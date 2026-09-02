#!/usr/bin/env python3
"""Per-frame progress labels for BEHAVIOR-1K, straight from LeRobot parquet.

    phi_t = D * (1 - cumsum_final + cumsum_t)

`phi_t` is the number of goal units satisfied at frame `t`, on the assumption the
demo ends satisfied. Dividing by `D` gives normalised progress in [0, 1]. `D` is the
task's goal-unit count; it is **measured from the reward trace**, never predicted --
see README.md, warning 1.

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
DEFAULT_MAX_PHI0 = 0.05
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

def repair_negatives(rewards: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero negatives that would drive the running reward sum below its floor.

    The demo collector rolls the simulator back and replays; the reward function
    emits a debit for credit it never issued in this trace. Such a debit pushes the
    running sum below zero, which no real predicate flip can do -- you cannot
    un-satisfy a unit that was never satisfied. Those are dropped. A negative that
    leaves the sum at or above zero is a genuine predicate flipping back and is kept.

    Returns (repaired_rewards, n_dropped).
    """
    out = np.asarray(rewards, dtype=np.float64).copy()
    running = 0.0
    dropped = 0
    for i, v in enumerate(out):
        if v == 0.0:
            continue
        if v < 0.0 and running + v < -EPS:
            out[i] = 0.0
            dropped += 1
        else:
            running += v
    return out, dropped


# ---------------------------------------------------------------------------
# per-episode labels
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeLabels:
    progress: np.ndarray        # float64, normalised, [0, 1]
    satisfied_count: np.ndarray  # float64, raw goal units, [0, D]
    phi0: float                 # normalised progress at t=0
    phi0_units: float           # phi0 * D == goal units that never fired
    n_dropped_rewards: int      # rollback debits zeroed by repair
    monotone: bool
    valid: bool
    drop_reason: str | None


def episode_labels(rewards: np.ndarray, D: int, terminated_last: bool = True,
                   truncated_any: bool = False) -> EpisodeLabels:
    """Build progress labels for one episode and run the validation gates."""
    repaired, n_dropped = repair_negatives(rewards)
    cumsum = np.cumsum(repaired)
    progress = 1.0 - cumsum[-1] + cumsum if len(cumsum) else cumsum
    satisfied = progress * D

    phi0 = float(progress[0]) if len(progress) else float("nan")
    phi0_units = phi0 * D
    monotone = bool(np.all(np.diff(progress) >= -EPS)) if len(progress) > 1 else True

    reason = None
    if not terminated_last:
        reason = "not_terminated"
    elif truncated_any:
        reason = "truncated"
    elif not (-EPS <= phi0 <= 1.0 + EPS):
        reason = "phi0_out_of_range"
    elif abs(phi0_units - round(phi0_units)) > INTEGRAL_TOL:
        reason = "phi0_not_integral"
    elif phi0 >= 1.0 - EPS:
        # Progress is the constant 1.0: the episode terminated having credited nothing,
        # so the label would teach that a freshly-reset scene is already complete.
        # Passes range and integrality, so it needs its own gate.
        reason = "no_headroom"
    elif progress.min() < -EPS or progress.max() > 1.0 + EPS:
        reason = "progress_out_of_range"

    return EpisodeLabels(
        progress=np.clip(progress, 0.0, 1.0),
        satisfied_count=np.clip(satisfied, 0.0, D),
        phi0=phi0, phi0_units=phi0_units, n_dropped_rewards=n_dropped,
        monotone=monotone, valid=reason is None, drop_reason=reason,
    )


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
        if rec is None or pd.isna(r["D"]):
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
    return rmap


def check_task_usable(task: str, reward_map: dict,
                      max_phi0: float = DEFAULT_MAX_PHI0,
                      min_valid_rate: float = DEFAULT_MIN_VALID_RATE) -> dict:
    """Raise TaskRefused unless this task's reward instrumentation is trustworthy.

    The detector is phi0. phi0 * D is the number of goal units that never fired in a
    demo that reached the goal, so phi0 = 0 is evidence of complete instrumentation
    and a high phi0 is evidence the reward under-fires. See README.md, warning 2 --
    and warning 3, which is why this is necessary but not sufficient.
    """
    rec = reward_map.get(task)
    if rec is None:
        raise TaskRefused(task, "not in the reward map -- D has never been measured "
                                "for this task, and D must never be guessed")

    D = rec.get("D")
    if not D or rec.get("reward_instrumentation") == "none":
        raise TaskRefused(task, "no reward signal at all in the demos -- D is not "
                                "measurable, so no progress label can be built")

    phi0 = float(rec.get("phi0_mean", 0.0))
    evidence = (f"mean phi0 = {phi0:.4f} ({rec.get('measurement_scope', 'sample-28pct')}) "
                f"with D = {D}, so {phi0 * D:.1f} of {D} goal units never fire even in "
                f"demos that reach the goal; labels would encode a task already "
                f"{phi0:.0%} complete at t=0")

    instr = rec.get("reward_instrumentation")
    if instr not in ("ok", None):
        raise TaskRefused(task, f"reward_instrumentation = {instr!r} -- {evidence}")

    if phi0 > max_phi0:
        raise TaskRefused(task, f"reward instrumentation suspect -- {evidence}. "
                                f"Threshold is phi0 <= {max_phi0}")

    valid_rate = float(rec.get("valid_rate", 1.0))
    if valid_rate < min_valid_rate:
        raise TaskRefused(task, f"only {valid_rate:.1%} of episodes pass validation "
                                f"(threshold {min_valid_rate:.0%})")
    return rec


# ---------------------------------------------------------------------------
# denominator audit (README warning 3)
# ---------------------------------------------------------------------------

def denominator_status(task: str, D: int, phi0_mean: float,
                       census_path: os.PathLike | str | None = None) -> tuple[str, int | None]:
    """Compare D against the BDDL predicates that must flip. (status, must_flip).

    phi0 only shows firing relative to D. If D was itself collapsed below the true
    goal size, phi0 = 0 merely confirms D and the events agree and says nothing about
    whether D matches the goal -- a second, independent defect mode phi0 cannot see.
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
    if D * (1.0 - phi0_mean) < D - 1e-6:
        return "UNDER_FIRES", must_flip
    return "CONSISTENT", must_flip


# ---------------------------------------------------------------------------
# per-task build
# ---------------------------------------------------------------------------

def build_task_labels(task: str, ds: Dataset, reward_map: dict,
                      out_dir: os.PathLike | str,
                      max_episodes: int | None = None,
                      max_phi0: float = DEFAULT_MAX_PHI0,
                      min_valid_rate: float = DEFAULT_MIN_VALID_RATE,
                      refuse_collapsed: bool = False) -> dict:
    """Emit labels.parquet + manifest.json for one task. Raises TaskRefused."""
    rec = check_task_usable(task, reward_map, max_phi0, min_valid_rate)
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

    dstatus, must_flip = denominator_status(task, D, float(rec.get("phi0_mean", 0.0)))
    if refuse_collapsed and dstatus == "COLLAPSED":
        raise TaskRefused(task, f"denominator {dstatus} -- D = {D} but {must_flip} BDDL "
                                f"predicates must flip, so {must_flip - D} of the goal is "
                                f"invisible to the reward. phi0 cannot see this")

    frame = ds.rewards_frame(task_index)
    episodes = ds.episode_indices(task_index)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    parts, drop_reasons, phi0s = [], {}, []
    used, dropped, nonmonotone, rewards_repaired = 0, 0, 0, 0
    for ep in episodes:
        e = frame[frame.episode_index == ep]
        lab = episode_labels(
            e["next.reward"].to_numpy(np.float64), D,
            terminated_last=bool(e["next.terminated"].iloc[-1]),
            truncated_any=bool(e["next.truncated"].any()),
        )
        if not lab.valid:
            dropped += 1
            drop_reasons[lab.drop_reason] = drop_reasons.get(lab.drop_reason, 0) + 1
            continue
        used += 1
        phi0s.append(lab.phi0)
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
        "mean_phi0": float(np.mean(phi0s)) if phi0s else None,
        "max_phi0": float(np.max(phi0s)) if phi0s else None,
        "mean_phi0_units": float(np.mean(phi0s) * D) if phi0s else None,
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
    p.add_argument("--max-phi0", type=float, default=DEFAULT_MAX_PHI0)
    p.add_argument("--min-valid-rate", type=float, default=DEFAULT_MIN_VALID_RATE)
    p.add_argument("--refuse-collapsed", action="store_true",
                   help="also refuse tasks whose D is below the BDDL must-flip count")
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
                                    a.max_phi0, a.min_valid_rate, a.refuse_collapsed)
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
        print(f"ok       {task}: D={man['D']} eps={man['episodes_used']}"
              f"/{man['episodes_considered']} frames={man['frames']} "
              f"mean_phi0={man['mean_phi0']:.4f}{flag}")
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
