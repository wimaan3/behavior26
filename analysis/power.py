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

HOW sigma_w IS PARAMETERISED
----------------------------
Q is the fraction of goal predicates satisfied at episode end, so it moves in
steps of one credit unit. Model the outcome as a reliable part plus u marginal
units that are genuine coin-flips under simulator noise. Then the unit count is
k_reliable + Binomial(u, 0.5), and Q is that divided by the denominator:

    sigma_w = 0.5 * sqrt(u) * step,     u = f * D

where `step` is the size of one credit unit as a fraction of Q, read from the
shortlist's measured `max_step_frac`, and f is the fraction of the task's D
units that are marginal. Two regimes fall out of the same formula:

  D = 1 (binary).  step = 1, so sigma_w^2 = 0.25 * f, maximised at 0.25. There
  is no small-sigma regime: an instance either reliably flips its one predicate
  or it does not. This is the D=1 shortlist tier.

  D > 1 (graded).  step ~= 1/D, so sigma_w ~= 0.5 * sqrt(f/D) -- noise shrinks
  as sqrt(D) because a one-unit wobble moves Q by only 1/D. At D=10 and f=0.20
  that is 0.072 against 0.224 for a binary task at the same f.

Grading is only real if episodes actually occupy the intermediate levels rather
than jumping 0 -> 1. The shortlist measures that as `frac_intermediate`, printed
in the header below; treat a graded D with a low frac_intermediate as binary in
disguise.

A design mixing tasks of different D gets the mean of sigma_w^2 over its chosen
tasks, since Var(d_i) is per-unit and units are averaged with equal weight.

COST
----
    rollout_seconds = scene_load + eval_timeout_frames / fps

Scene load is per trial and does NOT scale with episode length, so a task with
rel_eval_cost 0.2 is nowhere near 0.2x the wall clock. Using
`eval_timeout_frames` is deliberately conservative: a policy that rarely
succeeds runs to timeout most of the time.

    python analysis/power.py --tier graded             # the tier we actually run
    python analysis/power.py --tasks-from configs/experiments/001-dev-loop.yaml
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


def _opt_float(row: dict, key: str) -> float | None:
    """A shortlist column that only exists post-relabel. None if absent or blank."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def load_tasks(path: pathlib.Path, tier: str = "primary",
               names: list[str] | None = None) -> list[dict]:
    """Rows from the shortlist, cheapest-first.

    `tier` filters by shortlist tier; `names` selects an explicit task list and
    overrides the tier, so a frozen config can drive this directly.
    """
    if not path.is_file():
        sys.exit(f"shortlist not found: {path}")
    wanted = set(names or [])
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if wanted:
                if row.get("task") not in wanted:
                    continue
            elif tier and row.get("tier") != tier:
                continue
            try:
                rows.append({
                    "task": row["task"],
                    "rel_eval_cost": float(row["rel_eval_cost"]),
                    "frames": float(row["eval_timeout_frames"]),
                    "D": int(row["D"]),
                    # Post-relabel columns. max_step_frac is the size of one credit
                    # unit as a fraction of Q; frac_intermediate is how much of the
                    # episode is actually spent at partial credit.
                    "max_step_frac": _opt_float(row, "max_step_frac"),
                    "frac_intermediate": _opt_float(row, "frac_intermediate"),
                })
            except (KeyError, ValueError):
                continue
    if wanted:
        missing = sorted(wanted - {r["task"] for r in rows})
        if missing:
            sys.exit(f"task(s) not on the shortlist {path.name}: {missing}")
    rows.sort(key=lambda r: r["frames"])
    return rows


def credit_step(task: dict) -> float:
    """Size of one credit unit as a fraction of Q.

    Measured (`max_step_frac`) when the shortlist carries it, else the 1/D
    idealisation -- they agree to within 2% on the graded tier, because the
    goal predicates carry equal weight.
    """
    step = task.get("max_step_frac")
    if step and step > 0:
        return step
    return 1.0 / max(task["D"], 1)


def sigma_w_for(task: dict, f: float) -> float:
    """Within-instance SD of per-rollout Q, with f of the task's D units marginal.

    sigma_w = 0.5 * sqrt(f * D) * step.  At D=1, step=1 this is the binary
    sqrt(0.25 f); at D=10 it is ~3.2x smaller, because one flipped unit moves Q
    by a tenth rather than by the whole thing.
    """
    return 0.5 * math.sqrt(max(f, 0.0) * task["D"]) * credit_step(task)


def pooled_sigma_w(tasks: list[dict], f: float) -> float:
    """RMS of sigma_w over the chosen tasks.

    Var(d_i) is per-unit and units enter the mean with equal weight, so the
    design's variance uses the MEAN of sigma_w^2, not the mean of sigma_w.
    """
    return math.sqrt(sum(sigma_w_for(t, f) ** 2 for t in tasks) / len(tasks))


def rollout_hours(frames: float, fps: float, scene_load_s: float) -> float:
    return (scene_load_s + frames / fps) / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", type=pathlib.Path, default=SHORTLIST)
    ap.add_argument("--tier", default="primary", help="shortlist tier to draw from ('' = all)")
    ap.add_argument("--tasks-from", type=pathlib.Path, default=None, metavar="YAML",
                    help="read the task list from an experiment config's `tasks:` key, "
                         "e.g. configs/experiments/001-dev-loop.yaml. Overrides --tier, so "
                         "the power table describes the design we actually froze.")
    ap.add_argument("--sigma-w", type=float, default=None,
                    help="measured within-instance SD of per-rollout Q")
    ap.add_argument("--marginal-fraction", type=float, default=None,
                    help="alternative to --sigma-w: fraction of each task's D goal units "
                         "that are coin-flips. Scaled by that task's credit step, so the "
                         "same f means less noise on a graded task than a binary one.")
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

    names = None
    if args.tasks_from:
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            sys.exit("--tasks-from needs PyYAML (pip install -r requirements-tools.txt)")
        cfg = yaml.safe_load(args.tasks_from.read_text()) or {}
        names = cfg.get("tasks") or None
        if not names:
            sys.exit(f"{args.tasks_from} has no non-empty `tasks:` list")

    tasks = load_tasks(args.shortlist, args.tier, names)
    if not tasks:
        sys.exit(f"no '{args.tier}' rows in {args.shortlist}")

    # A measured --sigma-w is used as-is. Otherwise noise is expressed as f, the
    # fraction of marginal goal units, and converted per task -- the same f means
    # different sigma_w on a D=1 task and a D=10 one.
    fixed_sigma = args.sigma_w
    if args.marginal_fraction is not None:
        f_values = [args.marginal_fraction]
        labels = [f"f={args.marginal_fraction:.2f}"]
        sigma_src = f"marginal fraction f={args.marginal_fraction:.2f}"
    elif fixed_sigma is not None:
        f_values = [None]
        labels = ["measured"]
        sigma_src = "measured"
    else:
        f_values = [0.05, 0.10, 0.20, 0.40]
        labels = [f"f={f:.2f}" for f in f_values]
        sigma_src = "UNMEASURED"

    source = (f"tasks from {args.tasks_from}" if names
              else f"tier={args.tier or 'all'}")
    print(f"shortlist  {args.shortlist.relative_to(REPO)}  {source}  "
          f"({len(tasks)} tasks)")
    cheapest = ", ".join(f"{t['task']} ({t['rel_eval_cost']:.2f}x)" for t in tasks[:4])
    print(f"cheapest   {cheapest}")
    d_values = sorted({t["D"] for t in tasks})
    if d_values == [1]:
        note = "   <- D=1 means per-rollout Q is BINARY"
    elif min(d_values) > 1:
        note = "   <- all graded: Q moves in steps of 1/D, not 0 -> 1"
    else:
        note = "   <- mixed binary and graded"
    print(f"D values   {d_values}{note}")

    # Grading is only real if episodes occupy the intermediate levels. Show the
    # measurement rather than trusting D.
    graded = [t for t in tasks if t["D"] > 1 and t["frac_intermediate"] is not None]
    if graded:
        print("grading    " + "; ".join(
            f"{t['task'][:34]} D={t['D']} step={credit_step(t):.3f} "
            f"intermediate={t['frac_intermediate']:.2f}" for t in graded[:4]))
        nominal = [t for t in graded if t["frac_intermediate"] < 0.2]
        if nominal:
            print("           !! " + ", ".join(t["task"] for t in nominal)
                  + " is graded in name only (frac_intermediate < 0.2) -- treat as binary.")
    rate = "spot" if args.usd_per_gpu_hour < ON_DEMAND_USD else "on-demand"
    print(f"cost model {args.fps} fps, {args.scene_load:.0f}s scene load, "
          f"${args.usd_per_gpu_hour:.2f}/GPU-hr ({rate}), 2 arms")
    print(f"budget     ${args.total_budget_usd:.0f} total for the project")
    print(f"test       paired, alpha={args.alpha}, power={args.power}, sigma_b={args.sigma_b}")
    print()

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
                # sigma_w depends on the chosen tasks' D, so it is computed per
                # design rather than once for the whole table.
                cells = " ".join(
                    f"{mde(units, fixed_sigma if f is None else pooled_sigma_w(chosen, f), args.sigma_b, m, args.alpha, args.power):>9.3f}"
                    for f in f_values
                )
                flag = " " if hours <= args.budget_hours else "!"
                print(f"{k:>7}x{n:<3}{flag}  {units:>4} {rollouts:>9} "
                      f"{hours:>8.1f} {usd:>7.0f} {cells}")
        print()

    print("Columns after USD are the minimum detectable dQ at that noise level.")
    print(f"'!' marks designs whose EVAL alone exceeds the {args.budget_hours:.0f} GPU-hr budget.")
    if sigma_src != "measured":
        print()
        print("f is the fraction of each task's D goal units that are coin-flips, so the")
        print("same column means different sigma_w per task:")
        for t in tasks[:4]:
            row = "  ".join(f"f={f:.2f} -> {sigma_w_for(t, f):.3f}" for f in f_values if f)
            print(f"  {t['task'][:34]:<36} D={t['D']:<3} {row}")
    if fixed_sigma is None and sigma_src == "UNMEASURED":
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
