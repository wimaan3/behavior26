#!/usr/bin/env python3
"""Rank BEHAVIOR-1K tasks as training candidates and audit their reward instrumentation.

Inputs (this directory)
    behavior1k_task_table.csv     28.4%-sample measurement, all 100 tasks
    full_corpus_lengths.csv       all 20,000 episode lengths (meta only)
    full_corpus_remeasure.csv     full 200-episode re-measurement, tasks pulled locally
Input (../census, committed)
    goal_census_detail.json       per-literal BDDL goal facts
Outputs
    task_shortlist.csv            ranked training candidates
    bddl_audit.csv                task-3 audit table

Selection criteria, in the stated priority order:
    1. phi0 == 0 in EVERY episode          -- full headroom, complete instrumentation
    2. short mean episode length           -- eval timeout is 1.5x the task's own mean
    3. valid_rate >= 0.70
    4. enough episodes
Plus the task-3 exclusion: a task whose denominator is COLLAPSED (D below the number
of BDDL predicates that must flip) is dropped from the primary tier entirely.

Where a task has been re-measured on all 200 episodes, those numbers win. The sample
is not safe for phi0: `installing_a_fax_machine` reads phi0 = 0 on its 62-episode
sample and 0.108 across all 200, with 43 episodes above zero.

Key identity:
    phi0 * D        ==  goal units that NEVER fired
    D * (1 - phi0)  ==  reward events actually observed
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CENSUS = HERE.parent / "census" / "goal_census_detail.json"

MIN_VALID_RATE = 0.70
MIN_EPISODES = 30
N_PRIMARY = 12
# Deliberate high-D picks: the phi0==0 set is almost all D=1-2, which is nearly a
# binary reward and gives a progress head next to nothing to learn.
MEDIUM_D = ["cook_bacon", "make_rose_centerpieces", "chop_an_onion"]


def load() -> pd.DataFrame:
    tt = pd.read_csv(HERE / "behavior1k_task_table.csv")
    lens = pd.read_csv(HERE / "full_corpus_lengths.csv")
    df = tt.merge(lens, left_on="task_name", right_on="task_name", suffixes=("", "_len"))

    df["source"] = "sample-28pct"
    df["n_episodes_measured"] = df["episodes"]
    for c in ("D", "phi0_mean", "phi0_frac_gt0", "valid_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["episodes_nonmonotonic"] = pd.NA

    remeasure_path = HERE / "full_corpus_remeasure.csv"
    if remeasure_path.exists():
        rm = pd.read_csv(remeasure_path).set_index("task_name")
        for name, r in rm.iterrows():
            m = df.task_name == name
            if not m.any() or pd.isna(r["D"]):
                continue
            df.loc[m, ["D", "phi0_mean", "phi0_frac_gt0", "valid_rate",
                       "n_episodes_measured", "episodes_nonmonotonic", "source"]] = [
                r["D"], r["phi0_mean"], r["phi0_frac_gt0"], r["valid_rate"],
                r["n_episodes"], r["episodes_nonmonotonic"], "full-200ep"]
    return df


def audit(row, census) -> tuple[int, int, float, str]:
    """(must_flip, n_literals, reward_events_observed, denominator_status)."""
    lits = census[row.task_name]
    # initially_true is True / False / None(unknown); count anything not proven
    # already-satisfied as a predicate that must flip.
    must_flip = sum(1 for l in lits if l["initially_true"] is not True)
    D = int(row.D)
    events = D * (1.0 - float(row.phi0_mean))
    if D < must_flip:
        status = "COLLAPSED"       # denominator smaller than the goal it represents
    elif events < D - 1e-6:
        status = "UNDER_FIRES"     # denominator right, but units never credited
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
        rows.append({
            "task": r.task_name, "task_index": int(r.task), "D": int(r.D),
            "mean_episode_len": round(r.mean_len),
            "eval_timeout_frames": int(r.eval_timeout_frames),
            "rel_eval_cost": round(r.mean_len / corpus_mean, 2),
            "valid_rate": round(float(r.valid_rate), 3),
            "episodes_total": int(r.n_episodes),
            "episodes_measured": int(r.n_episodes_measured),
            "phi0_mean": round(float(r.phi0_mean), 4),
            "phi0_frac_gt0": round(float(r.phi0_frac_gt0), 4),
            "episodes_nonmonotonic": (None if pd.isna(r.episodes_nonmonotonic)
                                      else int(r.episodes_nonmonotonic)),
            "must_flip_predicates": mf, "n_goal_literals": nl,
            "reward_events_observed": round(ev, 2),
            "denominator_status": status,
            "reward_instrumentation": r.reward_instrumentation,
            "measurement": r.source,
        })
    allrows = pd.DataFrame(rows)
    allrows.sort_values(["denominator_status", "mean_episode_len"]).to_csv(
        HERE / "bddl_audit.csv", index=False)

    eligible = allrows[(allrows.phi0_frac_gt0 == 0.0)
                       & (allrows.valid_rate >= MIN_VALID_RATE)
                       & (allrows.episodes_measured >= MIN_EPISODES)]
    excluded = eligible[eligible.denominator_status == "COLLAPSED"]
    primary = (eligible[eligible.denominator_status != "COLLAPSED"]
               .sort_values("mean_episode_len").head(N_PRIMARY).copy())
    primary.insert(0, "tier", "primary")
    primary.insert(0, "rank", range(1, len(primary) + 1))

    medium = allrows[allrows.task.isin(MEDIUM_D)].copy()
    medium.insert(0, "tier", "medium_D")
    medium.insert(0, "rank", pd.NA)

    exc = excluded.copy()
    exc.insert(0, "tier", "excluded_collapsed_denominator")
    exc.insert(0, "rank", pd.NA)

    out = pd.concat([primary, medium[~medium.task.isin(primary.task)], exc], ignore_index=True)
    out.to_csv(HERE / "task_shortlist.csv", index=False)

    print(f"corpus mean episode length = {corpus_mean:.0f} over {int(allrows.episodes_total.sum())} episodes")
    print(f"phi0==0 & valid>=70% & >={MIN_EPISODES} eps : {len(eligible)} tasks")
    print(f"  of which COLLAPSED (excluded)            : {len(excluded)} "
          f"({', '.join(excluded.task)})")
    print(f"wrote task_shortlist.csv ({len(out)} rows) and bddl_audit.csv ({len(allrows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
