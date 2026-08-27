"""
Tag rollouts by failure signature so you don't have to watch every video.

The problem this solves: a dev-loop sweep of 36 rollouts is ~3.5 hours of MP4 if you
watch all of it. You will stop doing that by the second iteration. Tagging lets you watch
two videos per cluster instead of all of them.

These are heuristics over the rollout JSON fields only -- no video decoding. They are
deliberately crude and the thresholds are guesses until calibrated against real rollouts.
Tune `Thresholds` after you have watched a handful and confirmed the labels match.

Taxonomy
--------
    SOLVED              q == 1
    PARTIAL             0 < q < 1
    IMMOBILE            q == 0, base barely moved -- stuck at spawn, planner failure
    NO_MANIPULATION     q == 0, navigated but arms barely moved -- never found/reached target
    ACTIVE_NO_PROGRESS  q == 0, moved and manipulated, still nothing -- grasp/precision failure
    TIMEOUT             ran to the step limit
    CRASHED             no usable result

Usage
-----
    python -m analysis.failures rollouts/000-baseline-smoke
    python -m analysis.failures rollouts/exp --csv tagged.csv --sample 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.parse import load_rollouts


@dataclass
class Thresholds:
    """Calibrate these against real rollouts before trusting the labels."""

    base_moved_m: float = 1.0       # below this, the robot effectively never relocated
    arm_moved_m: float = 0.5        # summed L+R end-effector displacement
    timeout_ratio: float = 0.98     # steps / max_steps above this counts as a timeout


def tag_row(row: pd.Series, th: Thresholds) -> str:
    q = row.get("q_score")
    if q is None or pd.isna(q):
        return "CRASHED"

    if q >= 1.0:
        return "SOLVED"
    if q > 0:
        return "PARTIAL"

    base = row.get("dist_base") or 0.0
    arms = (row.get("dist_left") or 0.0) + (row.get("dist_right") or 0.0)

    if base < th.base_moved_m and arms < th.arm_moved_m:
        return "IMMOBILE"
    if arms < th.arm_moved_m:
        return "NO_MANIPULATION"
    return "ACTIVE_NO_PROGRESS"


def tag(df: pd.DataFrame, th: Thresholds | None = None) -> pd.DataFrame:
    th = th or Thresholds()
    out = df.copy()
    out["failure_tag"] = out.apply(lambda r: tag_row(r, th), axis=1)
    return out


def report(df: pd.DataFrame, sample: int = 0) -> None:
    counts = df["failure_tag"].value_counts()
    total = len(df)

    print("\nfailure breakdown")
    print("-" * 52)
    for tag_name, n in counts.items():
        bar = "#" * int(30 * n / total)
        print(f"  {tag_name:<20} {n:>4}  {100*n/total:>5.1f}%  {bar}")

    zero = df[df["q_score"] == 0]
    if len(zero):
        print(f"\n  {len(zero)}/{total} rollouts scored zero.")
        print("  Attack the largest non-SOLVED bucket first -- that is where the points are.")

    if sample:
        print(f"\nsample videos ({sample} per tag):")
        for tag_name in counts.index:
            if tag_name == "SOLVED":
                continue
            subset = df[df["failure_tag"] == tag_name].head(sample)
            print(f"\n  {tag_name}:")
            for _, r in subset.iterrows():
                print(f"    {r['task']}  inst={r['instance_id']}  q={r['q_score']:.2f}  "
                      f"({r['file']})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Tag BEHAVIOR rollouts by failure signature")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=2,
                    help="print N example rollouts per tag to go watch")
    ap.add_argument("--by-task", action="store_true", help="cross-tab tags against tasks")
    args = ap.parse_args()

    tagged = tag(load_rollouts(args.output_dir))
    report(tagged, args.sample)

    if args.by_task:
        print("\ntag x task:")
        print(pd.crosstab(tagged["task"], tagged["failure_tag"]).to_string())

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        tagged.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
