#!/usr/bin/env python3
"""Re-measure BEHAVIOR-1K reward instrumentation on the FULL episode set.

The committed `behavior1k_task_table.csv` / `behavior1k_episode_stats.csv` were
measured on the first parquet shard(s) of each task -- 5,677 of 20,000 episodes
(28.4%). D survives that sampling (it is a min over magnitudes) but phi0 and validity
do not: `installing_a_fax_machine` reads phi0 = 0 on its 62-episode sample and
phi0 = 0.108 across all 200.

Emits, into this directory:
  full_corpus_lengths.csv    all 100 tasks, from meta/episodes only (no bulk parquet)
  full_corpus_remeasure.csv  every task whose data/chunk-NNN is present locally

Usage:
    python -m analysis.reward.measure_corpus --data-root ~/behavior-data
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.reward import labels

HERE = pathlib.Path(__file__).resolve().parent


def corpus_lengths(ds: labels.Dataset) -> pd.DataFrame:
    frames = [pq.read_table(f, columns=["episode_index", "task_index", "length"]).to_pandas()
              for f in sorted(glob.glob(str(ds.root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))]
    m = pd.concat(frames, ignore_index=True)
    g = m.groupby("task_index").length
    out = pd.DataFrame({
        "task_index": g.mean().index,
        "n_episodes": g.count().values,
        "mean_len": g.mean().values.round(1),
        "median_len": g.median().values,
        "p90_len": g.quantile(0.9).values.round(1),
        "min_len": g.min().values,
        "max_len": g.max().values,
    })
    out.insert(1, "task_name", [ds.index_to_name[i] for i in out.task_index])
    out["eval_timeout_frames"] = (1.5 * out.mean_len).round().astype(int)
    out["rel_eval_cost"] = (out.mean_len / out.mean_len.mean()).round(3)
    return out.sort_values("mean_len").reset_index(drop=True)


def remeasure(ds: labels.Dataset, task_index: int) -> dict:
    name = ds.index_to_name[task_index]
    D = ds.measure_D(task_index)
    if D is None:
        return {"task_index": task_index, "task_name": name, "D": None,
                "n_episodes": len(ds.episode_indices(task_index)),
                "note": "no reward signal"}
    s0_prior, n_unknown_init, alignment = labels.initial_satisfied_bounds(name, D)
    frame = ds.rewards_frame(task_index)
    phis, reasons = [], {}
    hist_phis, struct = [], []
    never, lost, dips = [], [], []
    n_valid = nonmono = repaired = short = 0
    for ep in ds.episode_indices(task_index):
        e = frame[frame.episode_index == ep]
        r = e["next.reward"].to_numpy(np.float64)
        term = bool(e["next.terminated"].iloc[-1])
        trunc = bool(e["next.truncated"].any())
        lab = labels.episode_labels(r, D, term, trunc, s0_units=s0_prior)
        # the historical terminal-anchored phi0, kept so the two are comparable and
        # so the null baseline can be recomputed without re-reading the corpus
        hist = labels.episode_labels(r, D, term, trunc, anchor="terminal")
        if hist.valid:
            hist_phis.append(hist.phi0)
        if not lab.valid:
            reasons[lab.drop_reason] = reasons.get(lab.drop_reason, 0) + 1
            continue
        n_valid += 1
        phis.append(lab.phi0)
        never.append(lab.never_credited_units)
        lost.append(lab.lost_at_end_units)
        dips.append(lab.max_dip_units)
        short += int(lab.final < 1.0 - 1e-6)
        struct.append(labels.progress_structure(lab.progress, D))
        repaired += lab.n_dropped_rewards
        nonmono += 0 if lab.monotone else 1
    n = len(ds.episode_indices(task_index))
    phis = np.array(phis) if phis else np.array([np.nan])
    hist_phis = np.array(hist_phis) if hist_phis else np.array([np.nan])
    st = {k: float(np.nanmean([x[k] for x in struct])) if struct else np.nan
          for k in ("frac_intermediate", "max_step_frac", "n_credit_events",
                    "n_levels", "t_first_credit", "mean_progress")}
    return {
        "task_index": task_index, "task_name": name, "D": D,
        "n_magnitudes": len(ds.reward_magnitudes(task_index)),
        "magnitudes_multiple_of_unit": ds.magnitudes_are_multiples_of_unit(task_index, D),
        "n_episodes": n, "n_valid": n_valid, "valid_rate": round(n_valid / n, 4),
        # terminal anchor -- what the committed sweep and the shortlist were built on
        "phi0_mean": round(float(np.nanmean(hist_phis)), 6),
        "phi0_max": round(float(np.nanmax(hist_phis)), 6),
        "episodes_phi0_gt0": int((hist_phis > 1e-6).sum()),
        "phi0_frac_gt0": round(float((hist_phis > 1e-6).mean()), 4),
        # init anchor
        "phi0_init_mean": round(float(np.nanmean(phis)), 6),
        "initially_true_literals": s0_prior,
        "initially_true_unknown": n_unknown_init,
        "census_alignment": alignment,
        "never_credited_mean": round(float(np.nanmean(never)) if never else np.nan, 6),
        "never_credited_frac_gt0": round(float(np.mean(np.array(never) > 1e-6))
                                         if never else np.nan, 4),
        "lost_at_end_mean": round(float(np.nanmean(lost)) if lost else np.nan, 6),
        "max_dip_units": round(float(np.nanmax(dips)) if dips else np.nan, 3),
        "episodes_ended_short": short,
        "demo_completion_rate": round(1.0 - short / n_valid, 4) if n_valid else np.nan,
        # progress structure -- does the label carry a gradient at all
        "frac_intermediate": round(st["frac_intermediate"], 4),
        "max_step_frac": round(st["max_step_frac"], 4),
        "n_credit_events": round(st["n_credit_events"], 3),
        "n_levels": round(st["n_levels"], 3),
        "t_first_credit": round(st["t_first_credit"], 4),
        "mean_progress": round(st["mean_progress"], 4),
        "episodes_nonmonotonic": nonmono,
        "rollback_rewards_zeroed": repaired,
        "drop_reasons": json.dumps(reasons, sort_keys=True),
        "note": "",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default=str(HERE))
    a = p.parse_args(argv)

    ds = labels.Dataset(a.data_root)
    out_dir = pathlib.Path(a.out_dir)

    lengths = corpus_lengths(ds)
    lengths.to_csv(out_dir / "full_corpus_lengths.csv", index=False)
    print(f"full_corpus_lengths.csv: {len(lengths)} tasks, "
          f"{lengths.n_episodes.sum()} episodes, corpus mean {lengths.mean_len.mean():.0f}")

    present = sorted(int(pathlib.Path(d).name.split("-")[1])
                     for d in glob.glob(str(ds.root / "data" / "chunk-*")))
    rows = []
    for n, ti in enumerate(present, 1):
        rows.append(remeasure(ds, ti))
        # Dataset caches a task's reward frame; over 100 tasks that is the whole
        # corpus (~211M frames) resident at once. Nothing revisits a task, so drop it.
        ds._rewards.clear()
        if n % 10 == 0 or n == len(present):
            print(f"  remeasured {n}/{len(present)} tasks", flush=True)
    df = pd.DataFrame(rows).sort_values("task_index")
    df.to_csv(out_dir / "full_corpus_remeasure.csv", index=False)
    print(f"full_corpus_remeasure.csv: {len(df)} tasks with local parquet "
          f"({', '.join(df.task_name.head(6))}{' ...' if len(df) > 6 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
