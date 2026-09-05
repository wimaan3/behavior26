#!/usr/bin/env python3
"""Sample (28.4%) vs full-corpus (100%) measurement: what actually moved.

`D` is `1/min|reward|`. If a sampled episode set happens never to contain a
single-predicate transition, the smallest magnitude observed is `2/D` and the measured
denominator comes out wrong by a factor of two -- the failure that made
`picking_up_trash` wrong by 3x. That is a correctness risk, not a precision one, so
every task is re-measured over all 200 episodes and the two scopes are diffed here.

Reads   behavior1k_reward_map.json   (sample scope, 100 tasks)
        full_corpus_remeasure.csv    (full scope, whatever measure_corpus produced)
Writes  scope_delta.csv              per-task diff, and the defect list

Usage:
    python -m analysis.reward.compare_scopes
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

# The null baseline is the task-equal-weighted mean phi0 over tasks whose reward
# instrumentation is 'ok' -- the expected residual progress at t=0 that a predictor
# gets for free. Task-equal, not episode-equal: a 200-episode task must not outvote
# a 45-episode one when the question is "how defective is the corpus".
OK = "ok"


def load_sample() -> pd.DataFrame:
    m = json.loads((HERE / "behavior1k_reward_map.json").read_text())
    df = pd.DataFrame(m).T.reset_index().rename(columns={"index": "task_name"})
    for c in ("D", "phi0_mean", "valid_rate", "episodes", "n_valid"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def null_baseline(phi0: pd.Series, instrumentation: pd.Series) -> float:
    return float(phi0[instrumentation == OK].mean())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(HERE))
    a = p.parse_args(argv)

    s = load_sample()
    f = pd.read_csv(HERE / "full_corpus_remeasure.csv")
    df = s.merge(f, on="task_name", how="left", suffixes=("_sample", "_full"))

    df["D_changed"] = (df.D_sample.notna() & df.D_full.notna()
                       & (df.D_sample != df.D_full))
    df["phi0_delta"] = df.phi0_mean_full - df.phi0_mean_sample
    df["valid_delta"] = df.valid_rate_full - df.valid_rate_sample
    # A magnitude that is not an integer multiple of 1/D means D itself is wrong.
    df["integrality_ok"] = df.magnitudes_multiple_of_unit

    cols = ["task_index_sample", "task_name", "D_sample", "D_full", "D_changed",
            "episodes", "n_episodes", "phi0_mean_sample", "phi0_mean_full",
            "phi0_delta", "episodes_phi0_gt0", "phi0_frac_gt0",
            "valid_rate_sample", "valid_rate_full", "valid_delta",
            "integrality_ok", "reward_instrumentation", "drop_reasons", "note"]
    out = df[[c for c in cols if c in df.columns]].sort_values("task_index_sample")
    out.to_csv(pathlib.Path(a.out_dir) / "scope_delta.csv", index=False)

    n_full = int(df.D_full.notna().sum() + df.note.eq("no reward signal").sum())
    print(f"tasks re-measured at full scale : {n_full} / {len(df)}")
    print(f"episodes                        : sample {int(df.episodes.sum()):,}"
          f" -> full {int(df.n_episodes.sum()):,}")

    print("\n--- D ---")
    chg = df[df.D_changed]
    if len(chg):
        for _, r in chg.iterrows():
            print(f"  CHANGED {r.task_name}: D {r.D_sample:g} -> {r.D_full:g}")
    else:
        print(f"  unchanged on all {int((df.D_sample.notna() & df.D_full.notna()).sum())}"
              f" tasks measurable in both scopes")
    bad = df[df.integrality_ok == False]  # noqa: E712 -- NaN must not match
    print(f"  integrality (every |reward| a multiple of 1/D): "
          f"{'FAIL on ' + ', '.join(bad.task_name) if len(bad) else 'passes on all'}")

    print("\n--- null baseline (task-equal mean phi0, instrumentation ok) ---")
    b_s = null_baseline(df.phi0_mean_sample, df.reward_instrumentation)
    ok_full = df.phi0_mean_full.notna() & (df.reward_instrumentation == OK)
    b_f = float(df.phi0_mean_full[ok_full].mean())
    print(f"  sample : {b_s:.6f}  (n={int((df.reward_instrumentation == OK).sum())})")
    print(f"  full   : {b_f:.6f}  (n={int(ok_full.sum())})")

    print("\n--- phi0 moved most ---")
    mv = df[df.phi0_delta.abs() > 1e-6].sort_values("phi0_delta", key=abs, ascending=False)
    for _, r in mv.head(15).iterrows():
        print(f"  {r.task_name:42s} {r.phi0_mean_sample:.4f} -> {r.phi0_mean_full:.4f}"
              f"  ({r.phi0_delta:+.4f})  {int(r.episodes_phi0_gt0)}/200 eps under-fire")

    print("\n--- instrumentation defects at full scale ---")
    d = df[df.D_full.notna()].copy()
    d["events"] = d.D_full * (1 - d.phi0_mean_full)
    dead = df[df.note == "no reward signal"]
    for _, r in dead.iterrows():
        print(f"  NO SIGNAL   {r.task_name}: no reward at all in 200 episodes")
    for _, r in d[d.phi0_mean_full > 0.05].sort_values("phi0_mean_full", ascending=False).iterrows():
        print(f"  UNDER_FIRES {r.task_name:38s} D={int(r.D_full):2d} "
              f"phi0={r.phi0_mean_full:.4f} events={r.events:.2f}/{int(r.D_full)} "
              f"({int(r.episodes_phi0_gt0)}/200 eps)")
    marginal = d[(d.phi0_mean_full > 1e-6) & (d.phi0_mean_full <= 0.05)]
    for _, r in marginal.sort_values("phi0_mean_full", ascending=False).iterrows():
        print(f"  MARGINAL    {r.task_name:38s} D={int(r.D_full):2d} "
              f"phi0={r.phi0_mean_full:.4f} ({int(r.episodes_phi0_gt0)}/200 eps)")
    print(f"\nwrote scope_delta.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
