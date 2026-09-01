"""
Parse evaluator rollout JSONs into a flat table.

The evaluator writes one JSON per rollout under <output-dir>/json/. Fields, per the
challenge submission spec:

    task, instance_id, rollout_id, steps, success,
    agent_distance: {base, left, right},
    normalized_agent_distance,
    q_score: {final},
    time: {simulator_steps, simulator_time, normalized_time}

Q -- the fraction of BDDL goal predicates satisfied at episode end -- is the ranking
metric. Mean Q is averaged across all 100 tasks, so a task you never attempt scores zero
and still counts in the denominator. `summarize()` reports against a configurable task
universe for that reason.

Usage
-----
    python -m analysis.parse rollouts/000-baseline-smoke
    python -m analysis.parse rollouts/exp --csv out.csv --universe 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _dig(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_rollouts(output_dir: Path) -> pd.DataFrame:
    """Read every rollout JSON under <output_dir>/json/ (or output_dir itself)."""
    json_dir = output_dir / "json"
    if not json_dir.is_dir():
        json_dir = output_dir

    rows = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "timing_manifest.json":
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping {path.name}: {exc}")
            continue

        rows.append({
            "file": path.name,
            "task": d.get("task"),
            "instance_id": d.get("instance_id"),
            "rollout_id": d.get("rollout_id"),
            "steps": d.get("steps"),
            "success": bool(d.get("success", False)),
            # Scored as 0.0 when absent -- per the rules a missing result counts as zero.
            # But record separately THAT it was absent: a crashed rollout and a rollout
            # that legitimately achieved nothing both read 0.0, and they call for
            # completely different responses (fix the infra vs. fix the policy).
            "q_score": _dig(d, "q_score", "final", default=0.0),
            "q_missing": _dig(d, "q_score", "final") is None,
            "sim_steps": _dig(d, "time", "simulator_steps"),
            "sim_time": _dig(d, "time", "simulator_time"),
            "normalized_time": _dig(d, "time", "normalized_time"),
            "dist_base": _dig(d, "agent_distance", "base"),
            "dist_left": _dig(d, "agent_distance", "left"),
            "dist_right": _dig(d, "agent_distance", "right"),
            "normalized_agent_distance": d.get("normalized_agent_distance"),
        })

    if not rows:
        raise SystemExit(f"no rollout JSONs found under {json_dir}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, universe: int | None = None) -> dict:
    """Summary stats.

    `universe` is the number of tasks the leaderboard averages over (100 for a full
    submission). Pass it to see your true mean Q rather than the mean over attempted
    tasks only -- those differ enormously on a partial submission.
    """
    per_task = df.groupby("task")["q_score"].mean()
    attempted = len(per_task)
    n_tasks = universe or attempted

    total_q = per_task.sum()
    n_missing = int(df["q_missing"].sum()) if "q_missing" in df else 0
    return {
        "rollouts": len(df),
        # Non-zero here means some rollouts produced no score and are being counted as
        # zeros. That is an infrastructure number, not a policy number -- chase it.
        "rollouts_missing_q": n_missing,
        "tasks_attempted": attempted,
        "task_universe": n_tasks,
        "mean_q_over_attempted": round(per_task.mean(), 4),
        "mean_q_over_universe": round(total_q / n_tasks, 4),
        "success_rate": round(df["success"].mean(), 4),
        "tasks_nonzero_q": int((per_task > 0).sum()),
        "tasks_zero_q": int((per_task == 0).sum()),
        "mean_sim_time_s": round(df["sim_time"].mean(), 1) if df["sim_time"].notna().any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse BEHAVIOR rollout JSONs")
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--csv", type=Path, default=None, help="write the flat table here")
    ap.add_argument("--universe", type=int, default=None,
                    help="tasks the leaderboard averages over (100 for a full submission)")
    ap.add_argument("--per-task", action="store_true", help="print the per-task breakdown")
    args = ap.parse_args()

    df = load_rollouts(args.output_dir)
    stats = summarize(df, args.universe)

    print(f"\n{args.output_dir}")
    print("-" * 52)
    for k, v in stats.items():
        print(f"  {k:<26} {v}")

    if stats["rollouts_missing_q"]:
        print(f"\n  !! {stats['rollouts_missing_q']} rollout(s) have no q_score and are "
              "counted as zero.")
        print("  !! That is a crash or a truncated write, not a policy result. "
              "Check the job logs.")

    if args.universe and stats["tasks_attempted"] < args.universe:
        missing = args.universe - stats["tasks_attempted"]
        print(f"\n  note: {missing} task(s) unattempted and scored as zero.")
        print("  Partial submissions are allowed; missing instances count as zero.")

    if args.per_task:
        per_task = (df.groupby("task")
                      .agg(q=("q_score", "mean"),
                           success=("success", "mean"),
                           n=("q_score", "size"))
                      .sort_values("q", ascending=False))
        print("\nper task:")
        print(per_task.to_string(float_format=lambda x: f"{x:.3f}"))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
