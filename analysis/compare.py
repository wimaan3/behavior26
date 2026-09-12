"""
Paired comparison of two rollout sweeps.

Why paired
----------
The evaluator is nondeterministic, so a single sweep's mean Q carries real sampling
noise. Comparing two INDEPENDENT sweeps means the difference has to clear the noise of
both (below, at one rollout per instance and 36 instances):

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

The unit of analysis is one (task, instance) pair -- NOT one rollout
-------------------------------------------------------------------
When both arms run m seeds per instance, arm A's rollout 3 and arm B's rollout 3 are
two unrelated draws from a nondeterministic simulator. The rollout index carries no
correspondence between arms, so joining on it does not pair anything; it just lines up
noise with noise. It also miscounts: n instances x m seeds looks like n*m independent
pairs to a t-test that in truth has only n units, so the standard error comes out too
small by roughly sqrt(m) and the interval is narrower than the data earns.

So we collapse each arm to one number per instance first, and pair those:

    d_i = mean_m Q_B(i) - mean_m Q_A(i)

which is the quantity analysis/power.py sizes the experiment around. Seeds do their
work inside that mean -- they shrink the within-instance noise term sigma_w^2/m -- and
they never inflate n.

The catch: it only works if both runs cover identical (task, instance) keys. If they do
not, the pairing is broken and the whole advantage is gone -- so this module warns
loudly and, by default, refuses to report on a mismatched pair.

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

# The unit of analysis: what gets paired across arms.
PAIR_KEYS = ["task", "instance_id"]
# Identifies a single rollout WITHIN one arm. Used only to detect duplicates -- never
# to pair across arms; see the module docstring.
ROLLOUT_KEYS = PAIR_KEYS + ["rollout_id"]


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


def _collapse_to_units(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Collapse one arm's rollouts to one row per (task, instance).

    Q is averaged over the arm's seeds for that instance -- that mean IS the arm's
    estimate for the unit. `seeds` is kept so the report can say how much evidence
    stands behind each one.
    """
    dupes = df.duplicated(subset=ROLLOUT_KEYS).sum()
    if dupes:
        raise SystemExit(
            f"run {arm} has {dupes} duplicate {tuple(ROLLOUT_KEYS)} row(s). "
            "The same rollout appears twice, which would weight that instance's mean "
            "-- de-duplicate before comparing."
        )

    agg = {"q_score": ("q_score", "mean"), "seeds": ("q_score", "size")}
    if "success" in df:
        agg["success"] = ("success", "mean")
    if "q_missing" in df:
        # Rollouts that produced no score are counted as 0.0 in the mean. Carry the
        # count so a unit propped up by crashes is visible rather than silent.
        agg["q_missing"] = ("q_missing", "sum")
    return df.groupby(PAIR_KEYS, as_index=False).agg(**agg)


def _uniform_seeds(units: pd.DataFrame) -> int | None:
    """The seeds-per-unit count if every unit has the same one, else None."""
    counts = set(units["seeds"].tolist())
    return int(counts.pop()) if len(counts) == 1 else None


def load_pair(dir_a: Path, dir_b: Path) -> tuple[pd.DataFrame, dict]:
    """Load both sweeps, collapse each to (task, instance) units, and inner-join those.

    Returns (joined frame, coverage report). The coverage report is not decoration --
    read it before you read the number.
    """
    a = load_rollouts(dir_a)
    b = load_rollouts(dir_b)

    units_a = _collapse_to_units(a, "A")
    units_b = _collapse_to_units(b, "B")

    keys_a = set(map(tuple, units_a[PAIR_KEYS].values))
    keys_b = set(map(tuple, units_b[PAIR_KEYS].values))

    merged = units_a.merge(units_b, on=PAIR_KEYS, suffixes=("_a", "_b"), how="inner")
    merged["dq"] = merged["q_score_b"] - merged["q_score_a"]

    seeds_a = _uniform_seeds(units_a)
    seeds_b = _uniform_seeds(units_b)

    coverage = {
        "n_a": len(a),                     # rollouts read
        "n_b": len(b),
        "n_units_a": len(units_a),         # (task, instance) units
        "n_units_b": len(units_b),
        "n_paired": len(merged),           # units paired across both arms
        "seeds_per_unit_a": seeds_a,
        "seeds_per_unit_b": seeds_b,
        # False if either arm is ragged across its own units, or the arms disagree.
        "balanced_seeds": seeds_a is not None and seeds_a == seeds_b,
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

    Computed on the same (task, instance) UNITS the paired test uses -- each arm's
    per-instance mean over its seeds -- so the only difference between the two
    numbers is whether the pairing is used. Any gap is free resolution.

    Deliberately NOT computed on raw rollouts. With m seeds per instance, treating
    m draws from the same instance as m independent samples is pseudoreplication:
    it divides by n*m when there are only n independent units and returns an SE
    that is too small. That would be a smaller number than this one, but not a
    real baseline -- and comparing against it would understate what pairing bought
    while implying an unpaired design is cheaper than it is.
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
        print(f"  coverage: identical, {coverage['n_paired']} paired unit(s)")

    # Say what a "unit" is here, so n_pairs below is never mistaken for a rollout count.
    sa, sb = coverage["seeds_per_unit_a"], coverage["seeds_per_unit_b"]
    print(f"  rollouts: {coverage['n_a']} in A, {coverage['n_b']} in B "
          f"-> {coverage['n_paired']} (task, instance) unit(s)")
    if not coverage["balanced_seeds"]:
        print(f"  !! unbalanced seeds per unit (A={sa or 'ragged'}, B={sb or 'ragged'}).")
        print("  !! Units with fewer seeds carry more within-instance noise, so the")
        print("  !! equal-weight mean over units is no longer the efficient estimator.")
    elif sa and sa > 1:
        print(f"  seeds: {sa} per unit per arm, averaged within each arm before pairing")

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
            print(f"\n  resolution on these {stats['n_pairs']} unit(s):")
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
        cols = merged[PAIR_KEYS + ["seeds_a", "q_score_a", "seeds_b", "q_score_b", "dq"]]
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
        raise SystemExit("the two runs share no (task, instance_id) keys")

    report(merged, coverage, args.dir_a, args.dir_b,
           per_instance=args.per_instance, top=args.top, confidence=args.confidence)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}  ({len(merged)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
