#!/usr/bin/env python3
"""Power and cost for a PAIRED A/B, read off the real task shortlist.

We get two, maybe three A/B cycles. This exists so the design is chosen before
the first one rather than discovered by spending them.

THE MODEL
---------
Unit of analysis is one (task, instance) pair, evaluated in BOTH arms. With m
seeds per instance per arm:

    d_i   = mean_m Q_B(i) - mean_m Q_A(i)
    Var(d_i) = 2*sigma_w^2/m + sigma_b^2
    SE     = sqrt(Var(d_i) / N),        N = k tasks * n instances
    MDE    = (t_{1-a/2, N-1} + t_{power, N-1}) * SE

sigma_w  within-instance, within-arm SD of per-rollout Q. Pure simulator
         indeterminism -- the noise floor. UNKNOWN until measured; see
         `--sigma-w` and docs/AB_PROTOCOL.md section 2.
sigma_b  SD across instances of the TRUE per-instance effect. Pairing removes
         instance difficulty, not effect heterogeneity, so this does not shrink
         with m and sets a hard floor on what any budget can detect.

WHY sigma_w IS PROBABLY LARGE HERE
----------------------------------
Every primary task on the shortlist has D = 1: one predicate must flip, so
per-rollout Q is BINARY, not graded. For a binary outcome the within-instance
variance is pi*(1 - pi), maximised at 0.25. Parameterising by the fraction f of
instances that are genuine coin-flips under simulator noise:

    sigma_w^2 ~= 0.25 * f        f=0.10 -> 0.158,  f=0.20 -> 0.224

There is no small-sigma regime available: an instance either reliably flips the
predicate or it does not. Use --marginal-fraction to think in those units.

COST
----
    rollout_seconds = scene_load + eval_timeout_frames / fps

Scene load is per trial and does NOT scale with episode length, so a task with
rel_eval_cost 0.2 is nowhere near 0.2x the wall clock. Using
`eval_timeout_frames` is deliberately conservative: a policy that rarely
succeeds runs to timeout most of the time.

    python analysis/power.py                          # default sweep
    python analysis/power.py --sigma-w 0.22           # once measured
    python analysis/power.py --marginal-fraction 0.2
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SHORTLIST = REPO / "analysis" / "reward" / "task_shortlist.csv"

# From the challenge's published throughput and the repo's timing notes.
DEFAULT_FPS = 13.5
DEFAULT_SCENE_LOAD_S = 225.0     # midpoint of the published 150-300s
# Spot, not on-demand. The harness is resumable (--resume indexes completed
# rollouts from JSON contents), so a preemption costs the in-flight rollout and
# nothing else. Spot runs 50-70% below on-demand; we quote the pessimistic end.
DEFAULT_GPU_HOUR_USD = 0.50
ON_DEMAND_USD = 1.00


def _t_ppf(p: float, dof: int) -> float:
    """Two-sided t quantile; scipy when present, else normal + a small-N bump."""
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(p, dof))
    except ImportError:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(p)
        # Cornish-Fisher style correction; within ~1% of t for dof >= 8.
        return z + (z**3 + z) / (4 * dof) if dof > 0 else float("nan")


def mde(n_units: int, sigma_w: float, sigma_b: float, m: int,
        alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable dQ for a paired design."""
    if n_units < 2:
        return float("nan")
    var_d = 2.0 * sigma_w**2 / m + sigma_b**2
    se = math.sqrt(var_d / n_units)
    dof = n_units - 1
    return (_t_ppf(1 - alpha / 2, dof) + _t_ppf(power, dof)) * se


def load_tasks(path: pathlib.Path, tier: str = "primary") -> list[dict]:
    if not path.is_file():
        sys.exit(f"shortlist not found: {path}")
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if tier and row.get("tier") != tier:
                continue
            try:
                rows.append({
                    "task": row["task"],
                    "rel_eval_cost": float(row["rel_eval_cost"]),
                    "frames": float(row["eval_timeout_frames"]),
                    "D": int(row["D"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["frames"])
    return rows


def rollout_hours(frames: float, fps: float, scene_load_s: float) -> float:
    return (scene_load_s + frames / fps) / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", type=pathlib.Path, default=SHORTLIST)
    ap.add_argument("--tier", default="primary", help="shortlist tier to draw from ('' = all)")
    ap.add_argument("--sigma-w", type=float, default=None,
                    help="measured within-instance SD of per-rollout Q")
    ap.add_argument("--marginal-fraction", type=float, default=None,
                    help="alternative to --sigma-w: fraction of instances that are coin-flips")
    ap.add_argument("--sigma-b", type=float, default=0.05,
                    help="SD of the true per-instance effect (effect heterogeneity)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 3],
                    help="seeds per instance per arm to tabulate")
    ap.add_argument("--tasks", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--instances", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--scene-load", type=float, default=DEFAULT_SCENE_LOAD_S)
    ap.add_argument("--usd-per-gpu-hour", type=float, default=DEFAULT_GPU_HOUR_USD,
                    help=f"default {DEFAULT_GPU_HOUR_USD} = spot (pessimistic end of "
                         f"50-70%% off ${ON_DEMAND_USD:.2f} on-demand)")
    ap.add_argument("--total-budget-usd", type=float, default=200.0,
                    help="hard project budget; designs over it are flagged")
    ap.add_argument("--budget-hours", type=float, default=50.0,
                    help="mark designs that fit one A/B cycle's eval budget")
    args = ap.parse_args()

    tasks = load_tasks(args.shortlist, args.tier)
    if not tasks:
        sys.exit(f"no '{args.tier}' rows in {args.shortlist}")

    if args.marginal_fraction is not None:
        sigma_w = math.sqrt(0.25 * args.marginal_fraction)
        sigma_src = f"marginal fraction f={args.marginal_fraction:.2f}"
    elif args.sigma_w is not None:
        sigma_w = args.sigma_w
        sigma_src = "measured"
    else:
        sigma_w = None
        sigma_src = "UNMEASURED"

    print(f"shortlist  {args.shortlist.relative_to(REPO)}  tier={args.tier or 'all'}  "
          f"({len(tasks)} tasks)")
    cheapest = ", ".join(f"{t['task']} ({t['rel_eval_cost']:.2f}x)" for t in tasks[:4])
    print(f"cheapest   {cheapest}")
    d_values = sorted({t["D"] for t in tasks})
    print(f"D values   {d_values}"
          + ("   <- D=1 means per-rollout Q is BINARY" if d_values == [1] else ""))
    rate = "spot" if args.usd_per_gpu_hour < ON_DEMAND_USD else "on-demand"
    print(f"cost model {args.fps} fps, {args.scene_load:.0f}s scene load, "
          f"${args.usd_per_gpu_hour:.2f}/GPU-hr ({rate}), 2 arms")
    print(f"budget     ${args.total_budget_usd:.0f} total for the project")
    print(f"test       paired, alpha={args.alpha}, power={args.power}, sigma_b={args.sigma_b}")
    print()

    sigmas = [sigma_w] if sigma_w is not None else [
        math.sqrt(0.25 * f) for f in (0.05, 0.10, 0.20, 0.40)
    ]
    labels = ([sigma_src] if sigma_w is not None else
              [f"f={f:.2f}" for f in (0.05, 0.10, 0.20, 0.40)])

    for m in args.seeds:
        print(f"=== {m} seed{'s' if m > 1 else ''} per instance per arm ===")
        head = f"{'design':>14} {'N':>4} {'rollouts':>9} {'GPU-hr':>8} {'USD':>7} "
        head += " ".join(f"{lab:>9}" for lab in labels)
        print(head)
        print("-" * len(head))
        for k in args.tasks:
            if k > len(tasks):
                continue
            chosen = tasks[:k]                       # always the cheapest k
            per_arm_hours = sum(
                rollout_hours(t["frames"], args.fps, args.scene_load) for t in chosen
            )
            for n in args.instances:
                units = k * n
                rollouts = 2 * units * m
                hours = 2 * n * m * per_arm_hours
                usd = hours * args.usd_per_gpu_hour
                cells = " ".join(
                    f"{mde(units, s, args.sigma_b, m, args.alpha, args.power):>9.3f}"
                    for s in sigmas
                )
                flag = " " if hours <= args.budget_hours else "!"
                print(f"{k:>7}x{n:<3}{flag}  {units:>4} {rollouts:>9} "
                      f"{hours:>8.1f} {usd:>7.0f} {cells}")
        print()

    print("Columns after USD are the minimum detectable dQ at that noise level.")
    print(f"'!' marks designs whose EVAL alone exceeds the {args.budget_hours:.0f} GPU-hr budget.")
    if sigma_w is None:
        print()
        print("sigma_w IS NOT MEASURED. Every dQ column above is a hypothesis, not a")
        print("prediction. Run the noise-floor measurement in docs/AB_PROTOCOL.md first,")
        print("then re-run this with --sigma-w <measured>.")
    print()
    print(f"sigma_b={args.sigma_b} is a floor no budget removes: at m -> infinity the MDE")
    print(f"tends to {(_t_ppf(1 - args.alpha / 2, 99) + _t_ppf(args.power, 99)) * args.sigma_b:.3f}"
          f"/sqrt(N). Effect heterogeneity needs its own estimate from the first A/B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
