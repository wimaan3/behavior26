"""
Paired comparison of two rollout sweeps.

Why paired
----------
The evaluator is nondeterministic and we run one rollout per instance, so a single
sweep's mean Q carries real sampling noise. Comparing two INDEPENDENT 36-rollout sweeps
means the difference has to clear the noise of both:

    se_unpaired = sqrt(var_A/n_A + var_B/n_B)

which on a dev-loop-sized sweep needs roughly dQ > 0.115 before you can call it. That is
larger than the gap between 1st and 5th place in 2025 -- i.e. an unpaired A/B on this
sweep size cannot resolve the differences we actually care about.

Run both arms on the SAME instances and compare per instance instead, and the
instance-to-instance variance -- which is most of the total, because some instances are
simply harder -- cancels:

    d_i = q_B(i) - q_A(i)
    se_paired = sd(d) / sqrt(n)

That drops the detectable difference to roughly 0.033. Same GPU budget, ~3.5x the
resolution. This is the cheapest statistical win available to us, and it is why the dev
subset is frozen.

The catch: it only works if both runs cover identical (task, instance, rollout) keys.
If they do not, the pairing is broken and the whole advantage is gone -- so this module
warns loudly and, by default, refuses to report on a mismatched pair.

Usage
-----
    python -m analysis.compare rollouts/baseline rollouts/predicate-head
    python -m analysis.compare A B --per-instance --csv diff.csv
    python -m analysis.compare A B --allow-mismatch     # compare the intersection anyway
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from analysis.parse import load_rollouts

JOIN_KEYS = ["task", "instance_id", "rollout_id"]


def _t_critical(dof: int, confidence: float = 0.95) -> tuple[float, str]:
    """Two-sided critical value. Uses scipy when available, else a normal approximation.

    Lazy/optional import: scipy is not in requirements-tools.txt. The normal
    approximation is fine at n>=30 and slightly anti-conservative below that, so we say
    which one was used rather than quietly changing the meaning of the interval.
    """
    if dof < 1:
        return float("nan"), "undefined (n<2)"
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.ppf(0.5 + confidence / 2, dof)), f"t({dof})"
    except ImportError:
        from statistics import NormalDist
        return NormalDist().inv_cdf(0.5 + confidence / 2), "normal approx (scipy absent)"


def _p_value(t_stat: float, dof: int) -> float | None:
    try:
        from scipy import stats  # type: ignore
        return float(2 * stats.t.sf(abs(t_stat), dof))
    except ImportError:
        return None


def load_pair(dir_a: Path, dir_b: Path) -> tuple[pd.DataFrame, dict]:
    """Load both sweeps and inner-join on (task, instance_id, rollout_id).

    Returns (joined frame, coverage report). The coverage report is not decoration --
    read it before you read the number.
    """
    a = load_rollouts(dir_a)
    b = load_rollouts(dir_b)

    for name, df in (("A", a), ("B", b)):
        dupes = df.duplicated(subset=JOIN_KEYS).sum()
        if dupes:
            raise SystemExit(
                f"run {name} has {dupes} duplicate {tuple(JOIN_KEYS)} row(s). "
                "Pairing is ambiguous -- de-duplicate before comparing."
            )

    keys_a = set(map(tuple, a[JOIN_KEYS].values))
    keys_b = set(map(tuple, b[JOIN_KEYS].values))

    merged = a.merge(b, on=JOIN_KEYS, suffixes=("_a", "_b"), how="inner")
    merged["dq"] = merged["q_score_b"] - merged["q_score_a"]

    coverage = {
        "n_a": len(a),
        "n_b": len(b),
        "n_paired": len(merged),
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "identical_coverage": keys_a == keys_b,
    }
    return merged, coverage


def paired_stats(merged: pd.DataFrame, confidence: float = 0.95) -> dict:
    """Mean dQ, standard error, and a confidence interval on the paired difference."""
    d = merged["dq"].astype(float)
    n = len(d)
    if n == 0:
        raise SystemExit("no paired rollouts to compare")

    mean_d = float(d.mean())
    if n == 1:
        return {
            "n_pairs": 1, "mean_dq": round(mean_d, 4), "sd_dq": None, "se_dq": None,
            "ci_low": None, "ci_high": None, "ci_method": "undefined (n=1)",
            "t_stat": None, "p_value": None, "significant": False,
            "min_detectable_dq": None,
        }

    sd_d = float(d.std(ddof=1))
    se_d = sd_d / math.sqrt(n)
    crit, method = _t_critical(n - 1, confidence)
    half = crit * se_d
    t_stat = mean_d / se_d if se_d > 0 else float("inf") if mean_d else 0.0
    p = _p_value(t_stat, n - 1) if se_d > 0 else None

    return {
        "n_pairs": n,
        "mean_dq": round(mean_d, 4),
        "sd_dq": round(sd_d, 4),
        "se_dq": round(se_d, 4),
        "ci_low": round(mean_d - half, 4),
        "ci_high": round(mean_d + half, 4),
        "ci_method": method,
        "t_stat": round(t_stat, 3) if math.isfinite(t_stat) else None,
        "p_value": round(p, 5) if p is not None else None,
        # Significant iff the CI excludes zero.
        "significant": bool(se_d > 0 and abs(mean_d) > half),
        # The dQ this sweep could have detected -- the honest resolution of the run.
        "min_detectable_dq": round(half, 4),
    }


def unpaired_stats(merged: pd.DataFrame, confidence: float = 0.95) -> dict:
    """The same comparison done independently, to show what pairing bought.

    Computed on the SAME rollouts, so the only difference is whether the pairing is
    used. Any gap between the two min-detectable numbers is free resolution.
    """
    a = merged["q_score_a"].astype(float)
    b = merged["q_score_b"].astype(float)
    n = len(a)
    if n < 2:
        return {"se_dq": None, "min_detectable_dq": None}

    se = math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    crit, _ = _t_critical(2 * n - 2, confidence)
    return {
        "se_dq": round(se, 4),
        "min_detectable_dq": round(crit * se, 4),
    }


def report(merged: pd.DataFrame, coverage: dict, dir_a: Path, dir_b: Path,
           per_instance: bool = False, top: int = 5,
           confidence: float = 0.95) -> None:
    print(f"\nA: {dir_a}")
    print(f"B: {dir_b}")
    print("-" * 62)

    # -- coverage first. The number below means nothing if this is wrong. -------------
    if not coverage["identical_coverage"]:
        print("\n  !! COVERAGE MISMATCH -- the runs do not cover identical instances.")
        print(f"  !! {len(coverage['only_in_a'])} key(s) only in A, "
              f"{len(coverage['only_in_b'])} only in B.")
        for key in coverage["only_in_a"][:top]:
            print(f"  !!   only in A: {key}")
        for key in coverage["only_in_b"][:top]:
            print(f"  !!   only in B: {key}")
        print("  !! Pairing on the intersection discards the rest and can bias the")
        print("  !! comparison if what is missing is not missing at random.\n")
    else:
        print(f"  coverage: identical, {coverage['n_paired']} paired rollout(s)")

    stats = paired_stats(merged, confidence)
    unp = unpaired_stats(merged, confidence)

    mean_a = merged["q_score_a"].mean()
    mean_b = merged["q_score_b"].mean()
    print(f"\n  mean Q (A)          {mean_a:.4f}")
    print(f"  mean Q (B)          {mean_b:.4f}")
    print(f"  mean dQ (B - A)     {stats['mean_dq']:+.4f}")

    if stats["se_dq"] is not None:
        pct = int(confidence * 100)
        print(f"  std error (paired)  {stats['se_dq']:.4f}   [{stats['ci_method']}]")
        print(f"  {pct}% CI              [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]")
        if stats["p_value"] is not None:
            print(f"  p-value             {stats['p_value']:.5f}")
        verdict = ("SIGNIFICANT -- the interval excludes zero"
                   if stats["significant"]
                   else "not significant -- the interval spans zero")
        print(f"\n  {verdict}")

        if unp["min_detectable_dq"]:
            print(f"\n  resolution on these {stats['n_pairs']} rollouts:")
            print(f"    paired    could detect dQ >= {stats['min_detectable_dq']:.4f}")
            print(f"    unpaired  could detect dQ >= {unp['min_detectable_dq']:.4f}")
            if stats["min_detectable_dq"] and stats["min_detectable_dq"] > 0:
                gain = unp["min_detectable_dq"] / stats["min_detectable_dq"]
                print(f"    pairing bought {gain:.1f}x resolution for the same GPU spend")

    # -- per-task movers ---------------------------------------------------------------
    if top:
        by_task = (merged.groupby("task")["dq"].agg(["mean", "size"])
                   .sort_values("mean", ascending=False))
        movers = by_task[by_task["mean"] != 0]
        if len(movers):
            print(f"\n  biggest per-task moves (of {len(by_task)} task(s)):")
            for task, row in movers.head(top).iterrows():
                print(f"    {row['mean']:+.4f}  {task}  (n={int(row['size'])})")
            tail = movers.tail(top)
            for task, row in tail.iloc[::-1].iterrows():
                if task not in movers.head(top).index:
                    print(f"    {row['mean']:+.4f}  {task}  (n={int(row['size'])})")

    if per_instance:
        print("\n  per-instance differences:")
        cols = merged[JOIN_KEYS + ["q_score_a", "q_score_b", "dq"]]
        print(cols.sort_values("dq").to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired comparison of two BEHAVIOR rollout sweeps")
    ap.add_argument("dir_a", type=Path, help="baseline rollout dir")
    ap.add_argument("dir_b", type=Path, help="candidate rollout dir")
    ap.add_argument("--per-instance", action="store_true",
                    help="print every paired difference")
    ap.add_argument("--top", type=int, default=5, help="how many per-task movers to show")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--csv", type=Path, default=None, help="write the joined table here")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="compare the intersection even if coverage differs")
    args = ap.parse_args()

    merged, coverage = load_pair(args.dir_a, args.dir_b)

    if not coverage["identical_coverage"] and not args.allow_mismatch:
        report(merged, coverage, args.dir_a, args.dir_b, top=args.top)
        print("\nrefusing to report a paired result on mismatched coverage.")
        print("Re-run both arms on the same instances, or pass --allow-mismatch.")
        return 1

    if merged.empty:
        raise SystemExit("the two runs share no (task, instance_id, rollout_id) keys")

    report(merged, coverage, args.dir_a, args.dir_b,
           per_instance=args.per_instance, top=args.top, confidence=args.confidence)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  ({len(merged)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
