#!/usr/bin/env python3
"""Rank BEHAVIOR-1K tasks as training candidates and audit their reward instrumentation.

Inputs (this directory)
    behavior1k_task_table.csv     28.4%-sample measurement, all 100 tasks
    full_corpus_lengths.csv       all 20,000 episode lengths (meta only)
    full_corpus_remeasure.csv     full 200-episode re-measurement, both anchors
Input (../census, committed)
    goal_census_detail.json       per-literal BDDL goal facts
Outputs
    task_shortlist.csv            ranked training candidates
    bddl_audit.csv                task-3 audit table

Two things are being ranked, and they are not the same thing:

  TRUSTWORTHY   the reward can express the whole goal, and the labels built from it
                are correct. `denominator_status` + `demo_completion_rate`.
  GRADED        the label carries a gradient a progress head can learn from.
                `gradient_score` = frac_intermediate * (1 - max_step_frac).

`D` is not a proxy for the second. `cook_bacon` has D=7 and lands six of its seven
units in a single frame 1.5% before the end (`max_step_frac` 0.86), which is a step
function wearing a multi-unit denominator. Ranking on D alone selects for exactly the
tasks that cannot show whether shaping works.

Selection criteria for the primary tier, in the stated priority order:
    1. denominator_status == CONSISTENT   -- D matches the BDDL must-flip count and
                                             every valid demo reaches progress 1.0
    2. short mean episode length          -- eval timeout is 1.5x the task's own mean
    3. valid_rate >= 0.70, enough episodes
Every qualifying task is listed; the tier is not truncated.

The `graded` tier re-ranks the eligible pool (CONSISTENT or ENDS_SHORT, cost within
MAX_GRADED_REL_COST) by gradient instead of by cost. An ENDS_SHORT task is one whose
reward instrumentation is sound but whose demos stop a unit or two short of the goal;
under the init anchor those episodes are labelled with the progress they actually
reached, so they are trainable -- `demo_completion_rate` says how many finish.

Key identities, init-anchored:
    phi0_init_mean * D  ==  goal units already true at reset (a scene property)
    never_credited_mean ==  goal units the reward never credits anywhere
    demo_completion_rate == share of valid episodes whose progress reaches 1.0
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CENSUS = HERE.parent / "census" / "goal_census_detail.json"

MIN_VALID_RATE = 0.70
MIN_EPISODES = 30
# The primary tier lists EVERY task that qualifies, with no head() cut. A fixed cut is
# how `make_pizza` -- CONSISTENT, phi0 = 0 in 200 of 200, valid 0.995 -- came to be the
# thirteenth of thirteen qualifying tasks and appear nowhere in a twelve-row shortlist,
# while the README claimed all thirteen were in the primary tier.
N_GRADED = 8
# A task is worth listing as graded only if it is affordable to evaluate; the whole
# point of the shortlist is that eval cost is linear in the task's own mean length.
MAX_GRADED_REL_COST = 1.10

ANCHORED = ["phi0_init_mean", "never_credited_mean", "never_credited_frac_gt0",
            "lost_at_end_mean", "demo_completion_rate", "episodes_ended_short",
            "initially_true_literals", "initially_true_unknown", "census_alignment",
            "max_dip_units", "frac_intermediate", "max_step_frac", "n_credit_events",
            "n_levels", "t_first_credit", "mean_progress"]


def load() -> pd.DataFrame:
    tt = pd.read_csv(HERE / "behavior1k_task_table.csv")
    lens = pd.read_csv(HERE / "full_corpus_lengths.csv")
    df = tt.merge(lens, on="task_name", suffixes=("", "_len"))

    df["source"] = "sample-28pct"
    df["n_episodes_measured"] = df["episodes"]
    for c in ("D", "phi0_mean", "phi0_frac_gt0", "valid_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["episodes_nonmonotonic"] = pd.NA
    for c in ANCHORED:
        df[c] = pd.NA

    remeasure_path = HERE / "full_corpus_remeasure.csv"
    if remeasure_path.exists():
        rm = pd.read_csv(remeasure_path).set_index("task_name")
        cols = ["D", "phi0_mean", "phi0_frac_gt0", "valid_rate", "n_episodes",
                "episodes_nonmonotonic"] + [c for c in ANCHORED if c in rm.columns]
        for name, r in rm.iterrows():
            m = df.task_name == name
            if not m.any() or pd.isna(r["D"]):
                continue
            df.loc[m, [c for c in cols if c != "n_episodes"]] = [
                r[c] for c in cols if c != "n_episodes"]
            df.loc[m, ["n_episodes_measured", "source"]] = [r["n_episodes"], "full-200ep"]
    return df


def audit(row, census) -> tuple[int, int, float, str]:
    """(must_flip, n_literals, reward_events_observed, denominator_status)."""
    lits = census[row.task_name]
    # initially_true is True / False / None(unknown); count anything not proven
    # already-satisfied as a predicate that must flip.
    must_flip = sum(1 for l in lits if l["initially_true"] is not True)
    D = int(row.D)
    never = float(row.never_credited_mean) if pd.notna(row.never_credited_mean) else 0.0
    completion = (float(row.demo_completion_rate)
                  if pd.notna(row.demo_completion_rate) else None)
    events = D - never
    if D < must_flip:
        status = "COLLAPSED"        # denominator smaller than the goal it represents
    elif completion is None:
        status = "DEAD_UNITS" if never > 1e-6 else "CONSISTENT"
    elif completion <= 1e-9 and never > 1e-6:
        status = "DEAD_UNITS"       # not one demo ever shows the whole goal
    elif completion < 1.0 - 1e-9:
        status = "ENDS_SHORT"       # reward is fine, some demos stop a unit short
    else:
        status = "CONSISTENT"
    return must_flip, len(lits), events, status


def main() -> int:
    df = load()
    census = json.load(open(CENSUS))
    corpus_mean = df.mean_len.mean()

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.D):
            continue
        mf, nl, ev, status = audit(r, census)
        fi = float(r.frac_intermediate) if pd.notna(r.frac_intermediate) else float("nan")
        ms = float(r.max_step_frac) if pd.notna(r.max_step_frac) else float("nan")
        rows.append({
            "task": r.task_name, "task_index": int(r.task), "D": int(r.D),
            "mean_episode_len": round(r.mean_len),
            "eval_timeout_frames": int(r.eval_timeout_frames),
            "rel_eval_cost": round(r.mean_len / corpus_mean, 2),
            "valid_rate": round(float(r.valid_rate), 3),
            "episodes_total": int(r.n_episodes),
            "episodes_measured": int(r.n_episodes_measured),
            # --- init-anchored labelling facts ---
            "progress_offset_at_reset": (round(float(r.phi0_init_mean), 4)
                                         if pd.notna(r.phi0_init_mean) else None),
            "initially_true_literals": (int(r.initially_true_literals)
                                        if pd.notna(r.initially_true_literals) else None),
            "initially_true_unknown": (int(r.initially_true_unknown)
                                       if pd.notna(r.initially_true_unknown) else None),
            "census_alignment": r.census_alignment if pd.notna(r.census_alignment) else None,
            "never_credited_units": (round(float(r.never_credited_mean), 3)
                                     if pd.notna(r.never_credited_mean) else None),
            "lost_at_end_units": (round(float(r.lost_at_end_mean), 3)
                                  if pd.notna(r.lost_at_end_mean) else None),
            "demo_completion_rate": (round(float(r.demo_completion_rate), 3)
                                     if pd.notna(r.demo_completion_rate) else None),
            # --- progress structure ---
            "frac_intermediate": None if pd.isna(fi) else round(fi, 3),
            "max_step_frac": None if pd.isna(ms) else round(ms, 3),
            "n_levels": (round(float(r.n_levels), 2) if pd.notna(r.n_levels) else None),
            "t_first_credit": (round(float(r.t_first_credit), 3)
                               if pd.notna(r.t_first_credit) else None),
            "gradient_score": (None if (pd.isna(fi) or pd.isna(ms))
                               else round(fi * (1.0 - ms), 3)),
            "episodes_nonmonotonic": (None if pd.isna(r.episodes_nonmonotonic)
                                      else int(r.episodes_nonmonotonic)),
            "max_dip_units": (round(float(r.max_dip_units), 2)
                              if pd.notna(r.max_dip_units) else None),
            "must_flip_predicates": mf, "n_goal_literals": nl,
            "reward_events_observed": round(ev, 2),
            "denominator_status": status,
            "reward_instrumentation": r.reward_instrumentation,
            "measurement": r.source,
        })
    allrows = pd.DataFrame(rows)
    allrows.sort_values(["denominator_status", "mean_episode_len"]).to_csv(
        HERE / "bddl_audit.csv", index=False)

    usable = allrows[(allrows.demo_completion_rate > 0)
                     & (allrows.valid_rate >= MIN_VALID_RATE)
                     & (allrows.episodes_measured >= MIN_EPISODES)]
    excluded = usable[usable.denominator_status == "COLLAPSED"]
    eligible = usable[usable.denominator_status != "COLLAPSED"]

    primary = (eligible[eligible.denominator_status == "CONSISTENT"]
               .sort_values("mean_episode_len").copy())
    primary.insert(0, "tier", "primary")
    primary.insert(0, "rank", range(1, len(primary) + 1))

    graded = (eligible[eligible.rel_eval_cost <= MAX_GRADED_REL_COST]
              .sort_values("gradient_score", ascending=False).head(N_GRADED).copy())
    graded.insert(0, "tier", "graded")
    graded.insert(0, "rank", range(1, len(graded) + 1))

    exc = excluded.copy()
    exc.insert(0, "tier", "excluded_collapsed_denominator")
    exc.insert(0, "rank", pd.NA)

    out = pd.concat([primary, graded[~graded.task.isin(primary.task)], exc],
                    ignore_index=True)
    out.to_csv(HERE / "task_shortlist.csv", index=False)

    print(f"corpus mean episode length = {corpus_mean:.0f} over "
          f"{int(allrows.episodes_total.sum())} episodes")
    print(f"reward can express completion & valid>={MIN_VALID_RATE:.0%} & "
          f">={MIN_EPISODES} eps : {len(usable)} tasks")
    print(f"  of which COLLAPSED (excluded)                  : {len(excluded)}")
    print(f"denominator status over all {len(allrows)} measurable tasks: "
          + ", ".join(f"{k}={v}" for k, v in
                      allrows.denominator_status.value_counts().items()))
    flat = eligible[eligible.max_step_frac >= 0.5]
    print(f"eligible but effectively a step function (max_step >= 0.5): "
          f"{len(flat)} of {len(eligible)}")
    print(f"wrote task_shortlist.csv ({len(out)} rows) and "
          f"bddl_audit.csv ({len(allrows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
