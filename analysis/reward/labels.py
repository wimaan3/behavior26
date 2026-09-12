#!/usr/bin/env python3
"""Per-frame progress labels for BEHAVIOR-1K, straight from LeRobot parquet.

    satisfied_t = s0 + cumsum_t

`satisfied_t` is the number of goal units true at frame `t`. `s0` is the number
already true at reset, bounded by the BDDL :init block and pinned inside those bounds
by the deepest negative excursion of the reward trace. Dividing by `D` gives
normalised progress in [0, 1]. `D` is the task's goal-unit count; it is **measured
from the reward trace**, never predicted -- see README.md, warning 1.

The label is anchored at the START of the episode, not the end. The previous recipe,
`phi_t = D*(1 - cumsum_final + cumsum_t)`, assumed every demo finished the task and
shifted the curve until it did; a demo that ended one unit short was relabelled as
having started one unit ahead. It is still reachable as `anchor="terminal"` and is
what `behavior1k_episode_stats.csv` was measured with.

Emitted per task:
    <out>/<task>/labels.parquet   index, episode_index, frame_index, task_index,
                                  progress (float32 in [0,1]), satisfied_count (float32)
    <out>/<task>/manifest.json    D, episodes used/dropped, drop reasons, mean phi0,
                                  provenance

A task whose reward instrumentation is suspect is REFUSED: no labels are written and
the reason is recorded in <out>/<task>/REFUSED.json.

Usage:
    python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
        --tasks turning_on_radio vacuuming_floors
    python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
        --tasks-from analysis/reward/task_shortlist.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import json
import os
import pathlib
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Tolerances. Rewards are float32, so 1/D round-trips with ~1e-7 error; a D of 16
# leaves plenty of room under 1e-3 while still catching a genuinely wrong D.
EPS = 1e-6
INTEGRAL_TOL = 1e-3

# Refusal thresholds. phi0 is the fraction of goal units that never fired in a
# successful demo, so anything meaningfully above zero means the reward under-fired.
# A task is refused when NO episode in it ever reaches the whole goal: the reward can
# then never express task completion, whatever the policy does. That is the successor
# to the old `phi0 <= 0.05` gate, which under the terminal anchor could not tell a
# reward that under-fires from a demo that stopped a step early.
DEFAULT_MIN_DEMO_COMPLETION = 0.0
DEFAULT_MAX_PHI0 = None     # optional bound on the reset offset; off by default
DEFAULT_MIN_VALID_RATE = 0.70

REWARD_COLUMNS = ["index", "episode_index", "frame_index",
                  "next.reward", "next.terminated", "next.truncated"]


class TaskRefused(Exception):
    """Raised when a task's reward instrumentation is not fit to label."""

    def __init__(self, task: str, reason: str):
        self.task = task
        self.reason = reason
        super().__init__(f"{task}: {reason}")


# ---------------------------------------------------------------------------
# reward repair
# ---------------------------------------------------------------------------

def repair_negatives(rewards: np.ndarray, floor: float = 0.0) -> tuple[np.ndarray, int]:
    """Zero negatives that would drive the running reward sum below `floor`.

    The demo collector rolls the simulator back and replays; the reward function
    emits a debit for credit it never issued in this trace. Such a debit pushes the
    running sum below the number of goal units that were satisfied at reset, which no
    real predicate flip can do -- you cannot un-satisfy a unit that was never
    satisfied. Those are dropped. A negative that leaves the sum at or above the floor
    is a genuine predicate flipping back and is kept.

    `floor` is `-s0/D`, i.e. minus the fraction of the goal already true at reset.
    It defaults to 0, which is only correct for a task whose goal is entirely false
    at reset. Getting it wrong in either direction is a real defect:

      floor too high (0 when literals start true)
          the legitimate debit for breaking an initially-true literal is deleted, and
          the later credit for restoring it is kept -- the trace over-credits. This is
          what made `cook_bacon` read as 7 units of work when only 6 are.
      floor too low
          a rollback debit survives and the episode reads as a predicate flip-back
          that never happened.

    Returns (repaired_rewards, n_dropped).
    """
    out = np.asarray(rewards, dtype=np.float64).copy()
    running = 0.0
    dropped = 0
    for i, v in enumerate(out):
        if v == 0.0:
            continue
        if v < 0.0 and running + v < floor - EPS:
            out[i] = 0.0
            dropped += 1
        else:
            running += v
    return out, dropped


# ---------------------------------------------------------------------------
# how much of the goal is already true at reset
# ---------------------------------------------------------------------------

def initial_satisfied_bounds(task: str, D: int,
                             census_path: os.PathLike | str | None = None,
                             ) -> tuple[int, int, str]:
    """(s0_units, n_unknown, alignment) from the BDDL :init block.

    The reward trace cannot tell "this literal was true at reset and the robot broke
    it" from "the collector rolled the simulator back and re-issued credit". Only the
    BDDL says which, so `s0_units` is the count of goal literals the census marks
    `initially_true: True`.

    Literals marked `None` (unknown -- kinematic relations the `:init` block does not
    state) are NOT counted. Counting them would let a rollback debit masquerade as a
    broken initially-true literal, which on a D=1 task pins progress at a constant 1.0
    and throws the episode away: 104 of 200 `re_shelving_library_books` episodes died
    that way before this was tightened. They are returned as `n_unknown` instead,
    because they bound the residual risk in the other direction -- a literal that
    really is true at reset and gets broken will have its debit repaired away and its
    restoring credit kept, over-crediting by up to `n_unknown` units.

    The literal-to-unit mapping is only sound when `D` equals the number of goal
    literals. When it does not -- a quantifier expanded, or a collapsed denominator --
    no literal can be tied to a unit, so the prior is dropped to 0 and reported
    `ambiguous`.
    """
    if census_path is None:
        census_path = pathlib.Path(__file__).resolve().parent.parent / "census" / "goal_census_detail.json"
    census_path = pathlib.Path(census_path)
    if not census_path.exists():
        return 0, 0, "no_census"
    lits = json.loads(census_path.read_text()).get(task)
    if lits is None:
        return 0, 0, "no_census"
    n_true = sum(1 for l in lits if l["initially_true"] is True)
    n_unknown = sum(1 for l in lits if l["initially_true"] is None)
    if len(lits) != D:
        return 0, n_true + n_unknown, "ambiguous"
    return n_true, n_unknown, "aligned"


# ---------------------------------------------------------------------------
# per-episode labels
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeLabels:
    progress: np.ndarray        # float64, normalised, [0, 1]
    satisfied_count: np.ndarray  # float64, raw goal units, [0, D]
    phi0: float                 # normalised progress at t=0
    phi0_units: float           # phi0 * D == goal units already satisfied at reset
    n_dropped_rewards: int      # rollback debits zeroed by repair
    monotone: bool
    valid: bool
    drop_reason: str | None
    peak: float = float("nan")           # highest progress reached
    final: float = float("nan")          # progress on the last frame
    never_credited_units: float = float("nan")   # D * (1 - peak): units never satisfied
    lost_at_end_units: float = float("nan")      # D * (peak - final): earned then lost
    max_dip_units: float = 0.0           # largest single downward step, in goal units


def episode_labels(rewards: np.ndarray, D: int, terminated_last: bool = True,
                   truncated_any: bool = False, s0_units: int = 0,
                   anchor: str = "init") -> EpisodeLabels:
    """Build progress labels for one episode and run the validation gates.

    `anchor="init"` (the default) reads the reward trace forward from the goal state
    at reset:

        satisfied_t = s0 + cumsum_t

    where `s0` is the number of goal units already true at reset, bounded by the BDDL
    :init block (`initial_satisfied_bounds`) and pinned within those bounds by the
    deepest negative excursion of the trace. Progress is then whatever the episode
    actually reached -- it is not forced to end at 1.0.

    `anchor="terminal"` is the historical recipe, `phi_t = D*(1 - cumsum_final +
    cumsum_t)`, which assumes every demo ends with the goal satisfied and back-shifts
    the whole curve to make that true. It is retained only so the committed corpus
    sweep stays reproducible. It has two failure modes the init anchor does not:

      * a demo that ends one unit short is relabelled as though that unit had been
        satisfied at reset, so `phi0` reports missing instrumentation that is really a
        truncated demo (48 of 200 `chop_an_onion` episodes, and every task in the
        UNDER_FIRES band);
      * with `s0 > 0` it silently over-credits, because the debit for breaking an
        initially-true literal is repaired away while the credit for restoring it is
        kept.
    """
    if anchor not in ("init", "terminal"):
        raise ValueError(f"anchor must be 'init' or 'terminal', not {anchor!r}")
    if not 0 <= s0_units <= D:
        raise ValueError(f"s0_units {s0_units} outside [0, {D}]")

    if anchor == "terminal":
        repaired, n_dropped = repair_negatives(rewards)
        cumsum = np.cumsum(repaired)
        progress = 1.0 - cumsum[-1] + cumsum if len(cumsum) else cumsum
    else:
        # The census prior is a claim about the scene, not about this episode. Cap it
        # per episode at D - (units this trace credits), because a trace that credits
        # k units cannot have started with more than D - k already satisfied. This is
        # what catches a prior that is simply wrong: 44 of 200 cook_bacon episodes earn
        # credit for CLOSING the refrigerator without ever debiting for opening it, so
        # in those episodes the fridge was open at reset and the census's
        # `not open(...) initially_true` does not hold. Lowering s0 also lowers the
        # repair floor, which can change the trace, so iterate to a fixed point.
        s0 = float(s0_units)
        for _ in range(8):
            repaired, n_dropped = repair_negatives(rewards, floor=-s0 / D)
            cumsum = np.cumsum(repaired)
            capped = min(s0, D - float(cumsum.max()) * D) if len(cumsum) else s0
            capped = max(0.0, capped)
            if abs(capped - s0) < EPS:
                break
            s0 = capped
        s0_units = s0
        progress = s0_units / D + cumsum

    satisfied = progress * D
    phi0 = float(progress[0]) if len(progress) else float("nan")
    phi0_units = phi0 * D
    steps = np.diff(progress) if len(progress) > 1 else np.zeros(0)
    monotone = bool(np.all(steps >= -EPS))
    peak = float(progress.max()) if len(progress) else float("nan")
    final = float(progress[-1]) if len(progress) else float("nan")

    reason = None
    if not terminated_last:
        reason = "not_terminated"
    elif truncated_any:
        reason = "truncated"
    elif not (-EPS <= phi0 <= 1.0 + EPS):
        reason = "phi0_out_of_range"
    elif abs(phi0_units - round(phi0_units)) > INTEGRAL_TOL:
        reason = "phi0_not_integral"
    elif peak > 1.0 + EPS:
        # The trace credits more than D units. Under the init anchor this is the
        # rollback replay signature -- the same unit credited twice -- and it is a
        # corrupted trace, not a labelling choice. 3 of 200 cook_bacon episodes.
        reason = "over_credited"
    elif peak - float(progress.min()) < EPS:
        # The label is a constant for the whole episode -- the reward never fired, so
        # there is nothing to learn from it and, in a demo that is supposed to have
        # reached the goal, it is evidence the reward did not fire at all. Under the
        # terminal anchor this showed up as the constant 1.0 ("already complete at
        # reset"); under the init anchor it is the constant phi0. Either way it passes
        # range and integrality, so it needs its own gate.
        reason = "no_headroom"
    elif progress.min() < -EPS:
        reason = "progress_out_of_range"

    return EpisodeLabels(
        progress=np.clip(progress, 0.0, 1.0),
        satisfied_count=np.clip(satisfied, 0.0, D),
        phi0=phi0, phi0_units=phi0_units, n_dropped_rewards=n_dropped,
        monotone=monotone, valid=reason is None, drop_reason=reason,
        peak=peak, final=final,
        # clamped at 0: the gates already reject peak > 1, so anything below zero here
        # is float32 round-off and a negative "units never credited" reads as nonsense
        never_credited_units=max(0.0, D * (1.0 - peak)),
        lost_at_end_units=max(0.0, D * (peak - final)),
        max_dip_units=float(-steps.min() * D) if len(steps) and steps.min() < 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# progress structure -- does this task's label carry a gradient at all?
# ---------------------------------------------------------------------------

def progress_structure(progress: np.ndarray, D: int) -> dict:
    """Shape of one episode's progress curve.

    `D` counts goal units; it says nothing about whether they are reached one at a
    time. `cook_bacon` has D=7 and fires 6 of them in a single frame 1.5% before the
    end, which is a step function wearing a multi-unit denominator. These are the
    numbers that tell the two apart.
    """
    n = len(progress)
    if n == 0:
        return {}
    lo, hi = float(progress.min()), float(progress.max())
    steps = np.diff(progress) if n > 1 else np.zeros(0)
    up = steps[steps > EPS]
    rising = np.nonzero(progress > progress[0] + EPS)[0]
    return {
        # fraction of the episode spent strictly between the lowest and highest values
        # the episode reaches -- 0 for a step function, ~0.5+ for a genuine staircase.
        # Anchored on min/max rather than first/last so an episode that ends short is
        # still measured over the range it actually covered.
        "frac_intermediate": float(np.mean((progress > lo + EPS) & (progress < hi - EPS))),
        # largest single jump as a fraction of the WHOLE goal. cook_bacon scores 0.86:
        # six of its seven units land in one frame, so D=7 buys no gradient.
        "max_step_frac": float(up.max()) if len(up) else float("nan"),
        "n_credit_events": int(len(up)),
        "n_levels": int(len(np.unique(np.round(progress * D, 6)))),
        "t_first_credit": float(rising[0] / (n - 1)) if len(rising) and n > 1 else float("nan"),
        "mean_progress": float(progress.mean()),
    }


# ---------------------------------------------------------------------------
# local LeRobot dataset
# ---------------------------------------------------------------------------

class Dataset:
    """Read-only view of a local LeRobot v3.0 pull.

    Everything resolves from files on disk -- meta/episodes/ carries the
    episode -> data-file map -- so nothing here issues a network request. That is
    deliberate: the per-episode HF resolver is what trips the hub rate limit.
    """

    def __init__(self, root: os.PathLike | str):
        self.root = pathlib.Path(root).expanduser()
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"no LeRobot dataset at {self.root} (missing {info_path})")
        self.info = json.loads(info_path.read_text())
        self.name_to_index: dict[str, int] = {}
        self.index_to_name: dict[int, str] = {}
        with open(self.root / "meta" / "tasks.jsonl") as fh:
            for line in fh:
                rec = json.loads(line)
                self.name_to_index[rec["task_name"]] = rec["task_index"]
                self.index_to_name[rec["task_index"]] = rec["task_name"]
        self._ep_meta: dict[int, pd.DataFrame] = {}
        self._rewards: dict[int, pd.DataFrame] = {}

    # -- episode index ------------------------------------------------------
    def episode_meta(self, task_index: int) -> pd.DataFrame:
        """episode_index -> length, data chunk/file, for one task."""
        if task_index in self._ep_meta:
            return self._ep_meta[task_index]
        hits = []
        # meta/episodes is chunked by task in this dataset, but do not rely on it:
        # try the obvious chunk, then fall back to a scan.
        candidates = ([self.root / "meta" / "episodes" / f"chunk-{task_index:03d}"]
                      + sorted((self.root / "meta" / "episodes").glob("chunk-*")))
        seen = set()
        for d in candidates:
            if not d.is_dir() or d in seen:
                continue
            seen.add(d)
            for f in sorted(d.glob("file-*.parquet")):
                df = pq.read_table(f, columns=["episode_index", "task_index", "length",
                                               "data/chunk_index", "data/file_index"]).to_pandas()
                df = df[df.task_index == task_index]
                if len(df):
                    hits.append(df)
            if hits:
                break
        if not hits:
            raise FileNotFoundError(
                f"task_index {task_index} not found under {self.root/'meta'/'episodes'}")
        out = pd.concat(hits, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        self._ep_meta[task_index] = out
        return out

    def episode_indices(self, task_index: int) -> list[int]:
        return self.episode_meta(task_index).episode_index.tolist()

    def episode_lengths(self, task_index: int) -> pd.Series:
        m = self.episode_meta(task_index)
        return m.set_index("episode_index").length

    # -- frame data ---------------------------------------------------------
    def read_columns(self, task_index: int, columns: Sequence[str],
                     episode_indices: Iterable[int] | None = None) -> pd.DataFrame:
        """Read `columns` for a task, optionally restricted to some episodes."""
        meta = self.episode_meta(task_index)
        if episode_indices is not None:
            wanted = set(int(e) for e in episode_indices)
            meta = meta[meta.episode_index.isin(wanted)]
        need = list(dict.fromkeys(list(columns) + ["episode_index"]))
        frames = []
        for (chunk, fidx), _ in meta.groupby(["data/chunk_index", "data/file_index"]):
            path = self.root / "data" / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing shard {path} -- pull data/chunk-{chunk:03d}")
            frames.append(pq.read_table(path, columns=need).to_pandas())
        if not frames:
            return pd.DataFrame(columns=need)
        df = pd.concat(frames, ignore_index=True)
        df = df[df.episode_index.isin(set(meta.episode_index))]
        return df[[c for c in need if c in df.columns]]

    def rewards_frame(self, task_index: int) -> pd.DataFrame:
        """All reward/terminal columns for a task, cached. Small: no obs, no video."""
        if task_index not in self._rewards:
            df = self.read_columns(task_index, REWARD_COLUMNS)
            self._rewards[task_index] = df.sort_values(["episode_index", "frame_index"])
        return self._rewards[task_index]

    def episode_rewards(self, task_index: int, episode_index: int) -> np.ndarray:
        df = self.rewards_frame(task_index)
        e = df[df.episode_index == episode_index]
        if not len(e):
            raise KeyError(f"episode {episode_index} not in task {task_index}")
        return e["next.reward"].to_numpy(np.float64)

    # -- D ------------------------------------------------------------------
    def reward_magnitudes(self, task_index: int) -> list[float]:
        r = self.rewards_frame(task_index)["next.reward"].to_numpy(np.float64)
        nz = np.abs(r[r != 0.0])
        return sorted(set(np.round(nz, 6).tolist()))

    def measure_D(self, task_index: int) -> int | None:
        """D = 1 / min|reward| over the task. None if the task emits no reward."""
        mags = self.reward_magnitudes(task_index)
        if not mags:
            return None
        return int(round(1.0 / min(mags)))

    def magnitudes_are_multiples_of_unit(self, task_index: int, D: int) -> bool:
        return all(abs(m * D - round(m * D)) < INTEGRAL_TOL
                   for m in self.reward_magnitudes(task_index))


# ---------------------------------------------------------------------------
# reward map + refusal
# ---------------------------------------------------------------------------

def load_reward_map(path: os.PathLike | str,
                    remeasure: os.PathLike | str | None = None) -> dict:
    """Load the reward map, overlaying full-corpus measurements where they exist.

    The map is measured on the first parquet shard(s) of each task -- 28.4% of the
    corpus. `D` survives that; `phi0` and `valid_rate` do not.
    `installing_a_fax_machine` reads phi0 = 0 on its 62-episode sample and 0.1075
    across all 200, so a refusal gate reading the sample alone is blind to exactly
    the defect it exists to catch. full_corpus_remeasure.csv wins where present.
    """
    path = pathlib.Path(path)
    rmap = json.loads(path.read_text())
    for rec in rmap.values():
        rec.setdefault("measurement_scope", "sample-28pct")

    if remeasure is None:
        remeasure = path.parent / "full_corpus_remeasure.csv"
    remeasure = pathlib.Path(remeasure)
    if not remeasure.exists():
        return rmap

    for _, r in pd.read_csv(remeasure).iterrows():
        rec = rmap.get(r["task_name"])
        if rec is None:
            continue
        if pd.isna(r["D"]):
            # No measurable D even on the full corpus. The verdict is unchanged, but
            # its scope is not: "no reward signal in 200 of 200 episodes" is a much
            # stronger claim than "none in the 48 episodes the sample happened to hold",
            # and a manifest that still says sample-28pct understates the evidence.
            rec.update({"episodes": int(r["n_episodes"]),
                        "measurement_scope": "full-200ep"})
            continue
        rec.update({
            "D": int(r["D"]),
            "phi0_mean": float(r["phi0_mean"]),
            "phi0_frac_gt0": float(r["phi0_frac_gt0"]),
            "valid_rate": float(r["valid_rate"]),
            "episodes": int(r["n_episodes"]),
            "n_valid": int(r["n_valid"]),
            "measurement_scope": "full-200ep",
        })
        # init-anchored fields, present once measure_corpus.py has been re-run
        for col in ("never_credited_mean", "lost_at_end_mean", "phi0_init_mean",
                    "demo_completion_rate", "initially_true_literals",
                    "frac_intermediate", "max_step_frac", "t_first_credit"):
            if col in r.index and not pd.isna(r[col]):
                rec[col] = float(r[col])
    return rmap


def check_task_usable(task: str, reward_map: dict,
                      max_phi0: float | None = DEFAULT_MAX_PHI0,
                      min_valid_rate: float = DEFAULT_MIN_VALID_RATE,
                      min_demo_completion: float = DEFAULT_MIN_DEMO_COMPLETION) -> dict:
    """Raise TaskRefused unless this task's reward instrumentation is trustworthy.

    The detector is `demo_completion_rate`: the share of valid episodes whose progress
    reaches 1.0. A task where that is zero has no demo anywhere in its 200 that shows
    the whole goal satisfied, which is instrumentation failure, not a hard task --
    `putting_dishes_away_after_cleaning` fires exactly once out of D=14 in 200 of 200.

    What this deliberately does NOT refuse is a task where some demos finish and
    others stop a unit short. Under the init anchor those short episodes are labelled
    correctly with the progress they actually reached; the old gate read them as
    missing instrumentation and threw away most of the corpus with real gradient.

    `max_phi0`, if given, additionally bounds the progress offset at reset -- the
    fraction of the goal already true before the robot moves. It is off by default:
    the offset is a property of the scene, not a defect. `wash_dog_toys` starts at
    0.333 because two of its six literals are satisfied at reset and never broken.
    """
    rec = reward_map.get(task)
    if rec is None:
        raise TaskRefused(task, "not in the reward map -- D has never been measured "
                                "for this task, and D must never be guessed")

    D = rec.get("D")
    if not D or rec.get("reward_instrumentation") == "none":
        raise TaskRefused(task, "no reward signal at all in the demos -- D is not "
                                "measurable, so no progress label can be built")

    scope = rec.get("measurement_scope", "sample-28pct")
    instr = rec.get("reward_instrumentation")
    completion = rec.get("demo_completion_rate")
    never = rec.get("never_credited_mean")

    if instr not in ("ok", None):
        raise TaskRefused(task, f"reward_instrumentation = {instr!r} ({scope})")

    if completion is None:
        # pre-init-anchor map: fall back to the terminal-anchored phi0 so an old
        # reward map still refuses the reference defect rather than silently passing
        phi0 = float(rec.get("phi0_mean", 0.0))
        if phi0 > 0.05:
            raise TaskRefused(
                task, f"mean phi0 = {phi0:.4f} ({scope}) with D = {D}, so {phi0 * D:.1f} "
                      f"of {D} goal units are unaccounted for. This map predates the "
                      f"init anchor -- re-run measure_corpus.py for a real verdict")
    elif float(completion) <= min_demo_completion:
        raise TaskRefused(
            task, f"no demo reaches the whole goal: {float(completion):.1%} of valid "
                  f"episodes end at progress 1.0 ({scope}), with {float(never or 0):.2f} "
                  f"of {D} goal units never credited anywhere in the mean episode. The "
                  f"reward cannot express completion of this task")

    if max_phi0 is not None:
        offset = float(rec.get("phi0_init_mean", rec.get("phi0_mean", 0.0)))
        if offset > max_phi0:
            raise TaskRefused(
                task, f"progress starts at {offset:.3f} ({offset * D:.1f} of {D} goal "
                      f"units already true at reset), above the --max-phi0 bound "
                      f"{max_phi0}")

    valid_rate = float(rec.get("valid_rate", 1.0))
    if valid_rate < min_valid_rate:
        raise TaskRefused(task, f"only {valid_rate:.1%} of episodes pass validation "
                                f"(threshold {min_valid_rate:.0%})")
    return rec


# ---------------------------------------------------------------------------
# denominator audit (README warning 3)
# ---------------------------------------------------------------------------

def denominator_status(task: str, D: int, never_credited_mean: float,
                       demo_completion_rate: float | None = None,
                       census_path: os.PathLike | str | None = None) -> tuple[str, int | None]:
    """Compare D against the BDDL predicates that must flip. (status, must_flip).

    The under-firing arm is measured with units that are NEVER credited anywhere in
    the episode -- `D * (1 - peak)` -- not with terminal-anchored phi0. phi0 under the
    old anchor was the sum of two unrelated things, units the reward never credited
    and units the demo earned and then lost before it ended, and the second dominates:
    of the 39 tasks that read UNDER_FIRES with >85% of their must-flip count observed,
    only `wash_dog_toys` has units that never credit at all.

    The other blindness is unchanged: if D was itself collapsed below the true goal
    size, a clean firing record merely confirms D and the events agree and says
    nothing about whether D matches the goal.
    """
    if census_path is None:
        census_path = pathlib.Path(__file__).resolve().parent.parent / "census" / "goal_census_detail.json"
    census_path = pathlib.Path(census_path)
    if not census_path.exists():
        return "UNKNOWN", None
    census = json.loads(census_path.read_text())
    lits = census.get(task)
    if lits is None:
        return "UNKNOWN", None
    # initially_true is True / False / None(unknown); count anything not proven
    # already-satisfied as a predicate that must flip.
    must_flip = sum(1 for l in lits if l["initially_true"] is not True)
    if D < must_flip:
        return "COLLAPSED", must_flip
    if demo_completion_rate is None:
        return ("DEAD_UNITS" if never_credited_mean > 1e-6 else "CONSISTENT"), must_flip
    if demo_completion_rate <= 1e-9 and never_credited_mean > 1e-6:
        # not one demo in the corpus ever shows the whole goal
        return "DEAD_UNITS", must_flip
    if demo_completion_rate < 1.0 - 1e-9:
        # some demos finish, others stop short: the reward is fine, the demos truncate
        return "ENDS_SHORT", must_flip
    return "CONSISTENT", must_flip


# ---------------------------------------------------------------------------
# per-task build
# ---------------------------------------------------------------------------

def build_task_labels(task: str, ds: Dataset, reward_map: dict,
                      out_dir: os.PathLike | str,
                      max_episodes: int | None = None,
                      max_phi0: float | None = DEFAULT_MAX_PHI0,
                      min_valid_rate: float = DEFAULT_MIN_VALID_RATE,
                      refuse_collapsed: bool = False,
                      drop_incomplete: bool = False,
                      min_demo_completion: float = DEFAULT_MIN_DEMO_COMPLETION) -> dict:
    """Emit labels.parquet + manifest.json for one task. Raises TaskRefused."""
    rec = check_task_usable(task, reward_map, max_phi0, min_valid_rate,
                            min_demo_completion)
    D = int(rec["D"])
    task_index = ds.name_to_index.get(task, rec.get("task_index"))
    if task_index is None:
        raise TaskRefused(task, "no task_index in either the dataset or the reward map")

    measured_D = ds.measure_D(task_index)
    if measured_D != D:
        raise TaskRefused(task, f"D measured from the local pull is {measured_D} but the "
                                f"reward map says {D} -- refusing rather than guessing")
    if not ds.magnitudes_are_multiples_of_unit(task_index, D):
        raise TaskRefused(task, f"not every reward magnitude is an integer multiple of "
                                f"1/{D} -- D is wrong")

    dstatus, must_flip = denominator_status(
        task, D, float(rec.get("never_credited_mean") or 0.0),
        rec.get("demo_completion_rate"))
    s0_prior, n_unknown_init, alignment = initial_satisfied_bounds(task, D)
    if refuse_collapsed and dstatus == "COLLAPSED":
        raise TaskRefused(task, f"denominator {dstatus} -- D = {D} but {must_flip} BDDL "
                                f"predicates must flip, so {must_flip - D} of the goal is "
                                f"invisible to the reward. phi0 cannot see this")

    frame = ds.rewards_frame(task_index)
    episodes = ds.episode_indices(task_index)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    parts, drop_reasons, phi0s = [], {}, []
    never, lost, dips, structure = [], [], [], []
    used, dropped, nonmonotone, rewards_repaired, ended_short = 0, 0, 0, 0, 0
    for ep in episodes:
        e = frame[frame.episode_index == ep]
        lab = episode_labels(
            e["next.reward"].to_numpy(np.float64), D,
            terminated_last=bool(e["next.terminated"].iloc[-1]),
            truncated_any=bool(e["next.truncated"].any()),
            s0_units=s0_prior,
        )
        if not lab.valid:
            dropped += 1
            drop_reasons[lab.drop_reason] = drop_reasons.get(lab.drop_reason, 0) + 1
            continue
        short = lab.final < 1.0 - EPS
        if short and drop_incomplete:
            dropped += 1
            drop_reasons["incomplete_demo"] = drop_reasons.get("incomplete_demo", 0) + 1
            continue
        used += 1
        ended_short += int(short)
        phi0s.append(lab.phi0)
        never.append(lab.never_credited_units)
        lost.append(lab.lost_at_end_units)
        dips.append(lab.max_dip_units)
        structure.append(progress_structure(lab.progress, D))
        rewards_repaired += lab.n_dropped_rewards
        nonmonotone += 0 if lab.monotone else 1
        parts.append(pd.DataFrame({
            "index": e["index"].to_numpy(np.int64),
            "episode_index": np.int64(ep),
            "frame_index": e["frame_index"].to_numpy(np.int64),
            "task_index": np.int32(task_index),
            "progress": lab.progress.astype(np.float32),
            "satisfied_count": lab.satisfied_count.astype(np.float32),
        }))

    out_dir = pathlib.Path(out_dir)
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    if parts:
        table = pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False)
        pq.write_table(table, task_dir / "labels.parquet", compression="zstd")

    manifest = {
        "task": task,
        "task_index": int(task_index),
        "D": D,
        "D_source": "measured from the reward trace (1/min|reward|); cross-checked "
                    "against behavior1k_reward_map.json",
        "reward_magnitudes": ds.reward_magnitudes(task_index),
        "episodes_considered": len(episodes),
        "episodes_used": used,
        "episodes_dropped": dropped,
        "drop_reasons": drop_reasons,
        "anchor": "init",
        "mean_phi0": float(np.mean(phi0s)) if phi0s else None,
        "max_phi0": float(np.max(phi0s)) if phi0s else None,
        "mean_phi0_units": float(np.mean(phi0s) * D) if phi0s else None,
        "initially_true_literals": s0_prior,
        "initially_true_unknown": n_unknown_init,
        "census_alignment": alignment,
        "progress_offset_at_reset": (float(np.mean(phi0s)) if phi0s else None),
        "mean_never_credited_units": float(np.mean(never)) if never else None,
        "mean_lost_at_end_units": float(np.mean(lost)) if lost else None,
        "max_dip_units": float(np.max(dips)) if dips else None,
        "episodes_ended_short": ended_short,
        "demo_completion_rate": (1.0 - ended_short / used) if used else None,
        "progress_structure": {
            k: float(np.nanmean([st[k] for st in structure])) for k in
            ("frac_intermediate", "max_step_frac", "n_credit_events", "n_levels",
             "t_first_credit", "mean_progress")
        } if structure else None,
        "frames": int(sum(len(p) for p in parts)),
        "rollback_rewards_zeroed": rewards_repaired,
        "episodes_nonmonotonic": nonmonotone,
        "denominator_status": dstatus,
        "must_flip_predicates": must_flip,
        "measurement_scope": rec.get("measurement_scope"),
        "reward_instrumentation": rec.get("reward_instrumentation"),
        "corpus_valid_rate": rec.get("valid_rate"),
        "dataset_root": str(ds.root),
        "codebase_version": ds.info.get("codebase_version"),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _tasks_from_csv(path: os.PathLike | str) -> list[str]:
    df = pd.read_csv(path)
    col = "task" if "task" in df.columns else "task_name"
    return df[col].dropna().astype(str).tolist()


def main(argv: Sequence[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="local LeRobot pull (never a hub repo id)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--tasks-from", default=None, help="CSV with a `task` column")
    p.add_argument("--reward-map", default=str(here / "behavior1k_reward_map.json"))
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-phi0", type=float, default=DEFAULT_MAX_PHI0,
                   help="optional bound on the progress offset at reset. Off by "
                        "default -- the offset is a property of the scene, not a defect")
    p.add_argument("--min-demo-completion", type=float, default=DEFAULT_MIN_DEMO_COMPLETION,
                   help="refuse a task when at most this share of its valid episodes "
                        "reach progress 1.0 (default 0: refuse only when none do)")
    p.add_argument("--min-valid-rate", type=float, default=DEFAULT_MIN_VALID_RATE)
    p.add_argument("--refuse-collapsed", action="store_true",
                   help="also refuse tasks whose D is below the BDDL must-flip count")
    p.add_argument("--drop-incomplete-demos", action="store_true",
                   help="drop episodes that end below progress 1.0. Off by default: "
                        "under the init anchor such an episode is correctly labelled, "
                        "it just does not finish the task. Turn it on if the head is "
                        "trained to predict 1.0 at the end of every demo.")
    p.add_argument("--strict", action="store_true", help="abort on the first refusal")
    a = p.parse_args(argv)

    tasks = list(a.tasks or [])
    if a.tasks_from:
        tasks += _tasks_from_csv(a.tasks_from)
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        p.error("give --tasks and/or --tasks-from")

    ds = Dataset(a.data_root)
    rmap = load_reward_map(a.reward_map)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    emitted, refused = [], []
    for task in tasks:
        try:
            man = build_task_labels(task, ds, rmap, out, a.max_episodes,
                                    a.max_phi0, a.min_valid_rate, a.refuse_collapsed,
                                    a.drop_incomplete_demos, a.min_demo_completion)
        except TaskRefused as exc:
            refused.append({"task": task, "reason": exc.reason})
            d = out / task
            d.mkdir(parents=True, exist_ok=True)
            (d / "REFUSED.json").write_text(
                json.dumps({"task": task, "reason": exc.reason}, indent=2) + "\n")
            print(f"REFUSED  {task}: {exc.reason}", file=sys.stderr)
            if a.strict:
                return 2
            continue
        emitted.append(man)
        flag = "" if man["denominator_status"] == "CONSISTENT" else f"  [{man['denominator_status']}]"
        ps = man["progress_structure"] or {}
        print(f"ok       {task}: D={man['D']} eps={man['episodes_used']}"
              f"/{man['episodes_considered']} frames={man['frames']} "
              f"offset@reset={man['mean_phi0']:.4f} "
              f"complete={man['demo_completion_rate']:.0%} "
              f"gradient={ps.get('frac_intermediate', float('nan')):.2f}"
              f"/maxstep={ps.get('max_step_frac', float('nan')):.2f}{flag}")
        if man["denominator_status"] == "COLLAPSED":
            print(f"WARNING  {task}: D = {man['D']} but {man['must_flip_predicates']} BDDL "
                  f"predicates must flip -- part of the goal is invisible to the reward. "
                  f"Trainable, but it cannot demonstrate that shaping works.", file=sys.stderr)

    summary = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(ds.root),
        "tasks_emitted": [m["task"] for m in emitted],
        "tasks_refused": refused,
        "total_frames": sum(m["frames"] for m in emitted),
        "total_episodes": sum(m["episodes_used"] for m in emitted),
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{len(emitted)} task(s) emitted, {len(refused)} refused -> {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
