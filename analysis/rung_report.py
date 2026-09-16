"""Turn session-B rung logs into a results report. Pure Python.

Reads the timestamped logs written by scripts/session_b/rung2_5.sh
(`<unix-time> Step N: k=v, ...`) and applies the rules FIXED in AB_PROTOCOL
revision 2026-09-15 before the run -- not chosen after looking at the data.

BRANCH 0 (manipulation check), operationalised
    absent  arm B logged no progress_loss                          -> BROKEN
    NaN     any logged progress_loss is not finite                 -> BROKEN
    flat    mean(last W) is NOT below mean(first W) by more than
            2 standard errors (Welch), W = FLAT_WINDOW steps        -> BROKEN
    else    LEARNING
A fresh BCE head starts at ln 2 = 0.693; "learning" means it has left that level
by more than its own step-to-step noise, not that it has reached anything useful.

ACTION LOSS NOT DEGRADED (rung 2)
    Arm A and arm B see the same seed and the same batches, so their action_loss is
    compared PAIRED, step by step. Reported: mean paired difference B - A over the
    last W steps and its 95% interval. Degraded = interval entirely above 0.

THROUGHPUT (rung 4) from arm A only -- arm B pays two extra backward passes.
    steps/s = 1 / median step interval over steps >= STEADY_FROM (excludes compile).
    GPU utilisation median at steady state; low utilisation => data-bound.

TASK-COUNT DEPENDENCE (rung 5)
    two-task steps/s vs arm A's, same config and batch. Ratio reported.
"""
from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

FLAT_WINDOW = 100
STEADY_FROM = 50
BATCH = 32          # rung2_5.sh BATCH; only used to convert steps/s to samples/s
# tqdm writes CARRIAGE RETURNS, so a real line is
#     "<unix-time> \r\rStep 55: action_loss=..."
# and splitting the file on "\n" leaves the timestamp and the step text in the same
# physical line but separated by \r. Reading with Python's universal newlines would
# split them apart and the timestamp would be lost -- the first version of this
# parser found ZERO steps in every real log while passing on \r-free fixtures.
_TS = re.compile(r"^(\d+(?:\.\d+)?)\s")
_STEP = re.compile(r"Step (\d+): (.*)$")


@dataclass
class Run:
    steps: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    metrics: list[dict[str, float]] = field(default_factory=list)

    def series(self, key: str) -> list[float]:
        return [m[key] for m in self.metrics if key in m]


def read_log(path) -> str:
    """Read a training log WITHOUT newline translation.

    Python's text mode converts a lone "\r" into "\n", which splits
    "<timestamp> \r\rStep 55: ..." into two lines and strips every step of its
    timestamp. parse() then finds nothing. Reading a real 1000-step log through
    read_text() returned ZERO steps while \r-bearing strings passed in tests --
    the translation happens at read time, before the parser sees anything.
    """
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def parse(text: str) -> Run:
    run = Run()
    for raw in text.split("\n"):
        ts = _TS.match(raw)
        if not ts:
            continue
        # last \r-separated segment: tqdm repaints in place, so earlier segments on
        # the same line are superseded progress bars.
        m = _STEP.search(raw.split("\r")[-1].strip())
        if not m:
            continue
        vals = {}
        for part in m.group(2).split(","):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                try:
                    vals[k] = float(v)
                except ValueError:
                    vals[k] = float("nan")
        run.times.append(float(ts.group(1)))
        run.steps.append(int(m.group(1)))
        run.metrics.append(vals)
    return run


def _mean_se(xs):
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), float("inf")
    return statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def branch0(run_b: Run, window: int = FLAT_WINDOW) -> dict:
    p = list(by_step(run_b, "progress_loss").values())
    if not p:
        return {"verdict": "BROKEN", "reason": "absent: arm B logged no progress_loss"}
    if any(not math.isfinite(x) for x in p):
        return {"verdict": "BROKEN", "reason": "NaN/inf in progress_loss"}
    if len(p) < 2 * window:
        return {"verdict": "INCONCLUSIVE", "reason": f"only {len(p)} steps; need {2 * window} for the flat test"}
    m0, s0 = _mean_se(p[:window])
    m1, s1 = _mean_se(p[-window:])
    drop = m0 - m1
    se = math.sqrt(s0 ** 2 + s1 ** 2)
    z = drop / se if se > 0 else float("inf")
    verdict = "LEARNING" if z > 2 else "BROKEN"
    return {"verdict": verdict, "first_mean": m0, "last_mean": m1, "drop": drop, "z": z,
            "reason": "flat: not below the first window by >2 SE" if verdict == "BROKEN" else "decreasing by >2 SE"}


def by_step(run: Run, key: str) -> dict[int, float]:
    """step -> value, for rows that carry `key`.

    NOT zip(run.steps, run.series(key)): the calibration writes a SECOND line for
    the same step with different keys, so those two lists have different lengths
    and zipping them silently misaligns every later step.
    """
    return {s: m[key] for s, m in zip(run.steps, run.metrics) if key in m}


def paired_action(run_a: Run, run_b: Run, window: int = FLAT_WINDOW) -> dict:
    a = by_step(run_a, "action_loss")
    b = by_step(run_b, "action_loss")
    common = sorted(set(a) & set(b))[-window:]
    if len(common) < 2:
        return {"verdict": "INCONCLUSIVE", "n": len(common)}
    d = [b[s] - a[s] for s in common]
    m, se = _mean_se(d)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return {"n": len(d), "mean_diff_B_minus_A": m, "ci95": (lo, hi),
            "verdict": "DEGRADED" if lo > 0 else "NOT_DEGRADED"}


def throughput(run: Run, steady_from: int = STEADY_FROM) -> dict:
    pts = [(s, t) for s, t in zip(run.steps, run.times) if s >= steady_from]
    if len(pts) < 3:
        return {"steps_per_s": None, "n": len(pts)}
    dts = [(t2 - t1) / (s2 - s1) for (s1, t1), (s2, t2) in zip(pts, pts[1:]) if s2 > s1]
    med = statistics.median(dts)
    mean = statistics.fmean(dts)
    # Both, deliberately. Step time here is BIMODAL -- a fast step plus a periodic
    # loader stall -- so the median describes the fast path and the MEAN is what
    # sets wall clock and cost. Quoting only the median underprices a run.
    return {"steps_per_s": 1 / med if med > 0 else None, "s_per_step": med,
            "mean_s_per_step": mean, "p90_s_per_step": sorted(dts)[int(len(dts) * 0.9)],
            "wall_steps_per_s": 1 / mean if mean > 0 else None, "n": len(dts)}


def stalls(run: Run, steady_from: int = STEADY_FROM, threshold_s: float = 20.0) -> dict:
    """Locate the slow steps and ask whether they are PERIODIC.

    Rung 4 measured median 3.90 s/step but mean 8.11 s/step. The mean is what the
    clock charges, but on its own it does not say what to fix. The step numbers do:
    the slow steps landed on 72, 96, 120, ... -- exactly every 24 -- and 24 is the
    dataloader pipeline depth (8 workers x prefetch 3), not anything about the data.
    A constant period is the signature of a prefetch sawtooth: the trainer drains a
    full queue at the GPU-bound rate, then blocks while the workers refill it. That
    is fixed by adding workers; a slow *dataset* is not.

    `clock_share` is the fraction of wall time spent inside stalls -- the headline,
    because a stall on 4% of the steps can own half the run.
    """
    pts = [(s, t) for s, t in zip(run.steps, run.times) if s >= steady_from]
    if len(pts) < 3:
        return {"n": 0, "period": None, "clock_share": 0.0, "median_stall_s": None,
                "steps": [], "total_s": 0.0}
    iv = [(s2, (t2 - t1) / (s2 - s1)) for (s1, t1), (s2, t2) in zip(pts, pts[1:]) if s2 > s1]
    total = sum(d for _, d in iv)
    slow = [(s, d) for s, d in iv if d > threshold_s]
    if not slow:
        return {"n": 0, "period": None, "clock_share": 0.0, "median_stall_s": None,
                "steps": [], "total_s": total}
    gaps = [b - a for (a, _), (b, _) in zip(slow, slow[1:])]
    # "periodic" = every gap identical. Anything less regular is not a pipeline
    # artefact and must not be reported as one.
    period = gaps[0] if gaps and len(set(gaps)) == 1 else None
    return {"n": len(slow), "period": period,
            "clock_share": sum(d for _, d in slow) / total if total else 0.0,
            "median_stall_s": statistics.median([d for _, d in slow]),
            "steps": [s for s, _ in slow], "total_s": total}


def sustained_loader_rate(st: dict, fast_s_per_step: float, batch_size: int) -> float | None:
    """Samples/s the loader actually sustains, from the sawtooth geometry.

    One cycle delivers `period` batches and costs (period-1) fast steps plus one
    stall, so the rate is period*batch / cycle_seconds. Compare against
    batch/fast_s_per_step -- what a step consumes -- to size the worker count.
    """
    if not st.get("period") or not st.get("median_stall_s"):
        return None
    cycle = (st["period"] - 1) * fast_s_per_step + st["median_stall_s"]
    return st["period"] * batch_size / cycle if cycle > 0 else None


def gpu_util(csv_path: Path, t_from: float | None = None) -> float | None:
    if not csv_path.exists():
        return None
    vals = []
    for row in csv.reader(csv_path.open()):
        if len(row) >= 2:
            try:
                t, u = float(row[0]), float(row[1])
            except ValueError:
                continue
            if t_from is None or t >= t_from:
                vals.append(u)
    return statistics.median(vals) if vals else None


def report(log_dir: Path) -> str:
    """Markdown report for a rung2_5.sh output directory."""
    from analysis.lambda_calibration import calibrate, parse_log

    def load(name):
        p = log_dir / f"train_{name}.log"
        if not p.exists():
            return Run(), ""
        text = read_log(p)
        return parse(text), text

    a, _ = load("rung4_armA")
    b, b_text = load("rung23_armB")
    m, _ = load("rung5_2task")
    out = ["# Session B rungs 2-5 -- results", "",
           f"Rules fixed in AB_PROTOCOL revision 2026-09-15 before the run (window {FLAT_WINDOW}, "
           f"steady state from step {STEADY_FROM}).", ""]

    b0 = branch0(b)
    out += ["## Rung 2 -- Branch 0 manipulation check", "", f"**{b0['verdict']}** -- {b0['reason']}"]
    if "z" in b0:
        out.append(f"progress_loss first {FLAT_WINDOW}: {b0['first_mean']:.4f}, last {FLAT_WINDOW}: "
                   f"{b0['last_mean']:.4f}, drop {b0['drop']:+.4f}, z = {b0['z']:.1f}")
    pa = paired_action(a, b)
    out += ["", "## Rung 2 -- action loss with the head present (paired, same batches)", ""]
    if "ci95" in pa:
        lo, hi = pa["ci95"]
        out.append(f"**{pa['verdict']}** -- mean B - A over last {pa['n']} steps "
                   f"{pa['mean_diff_B_minus_A']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    else:
        out.append(f"**{pa['verdict']}** ({pa.get('n', 0)} common steps)")

    out += ["", "## Rung 3 -- lambda calibration (gradient share)", ""]
    try:
        rows = parse_log(b_text)
        for first in (0, STEADY_FROM, max(0, len(rows) - FLAT_WINDOW)):
            c = calibrate(rows, target_share=0.2, current_lambda=0.1, first_step=first)
            out.append(f"- steps >= {first:>4} (n={c.steps_used}): median |g_a|/|g_p| {c.ratio_median:.3f}; "
                       f"share at lambda=0.1 {c.share_at_current:.1%}; lambda for 10/20/30% = "
                       f"{0.1/0.9*c.ratio_median:.3f} / {0.25*c.ratio_median:.3f} / {0.3/0.7*c.ratio_median:.3f}"
                       + (f"; median cos {c.cosine_median:+.3f}" if c.cosine_median is not None else ""))
        out.append("")
        out.append("If these three windows disagree sharply, lambda is still drifting -- report that rather "
                   "than pin lambda to an early transient (AB_PROTOCOL session-B note).")
    except ValueError as e:
        out.append(f"not computable: {e}")

    ta, tm = throughput(a), throughput(m)
    t_from = a.times[a.steps.index(STEADY_FROM)] if STEADY_FROM in a.steps else None
    ua = gpu_util(log_dir / "gpu_rung4_armA.csv", t_from)
    out += ["", "## Rung 4 -- throughput (arm A)", ""]
    if ta["steps_per_s"]:
        out.append(f"median {ta['s_per_step']:.2f} s/step, **mean {ta['mean_s_per_step']:.2f} s/step** "
                   f"(p90 {ta['p90_s_per_step']:.2f}) over {ta['n']} intervals; "
                   f"median GPU utilisation {ua if ua is not None else 'n/a'}%")
        out.append("")
        out.append("Step time is bimodal: a fast step plus a periodic loader stall. Cost follows the MEAN.")
        for n_steps in (10_000, 30_000):
            h = n_steps * ta["mean_s_per_step"] / 3600
            out.append(f"- {n_steps:,} steps ~ {h:.1f} h ~ ${h * 0.72:.2f} per arm at $0.72/h")

        st = stalls(a)
        out += ["", "### Where the time goes", ""]
        if st["n"] == 0:
            out.append("No step exceeded 20 s -- step time is flat and the mean is the median.")
        else:
            out.append(f"{st['n']} intervals over 20 s (median {st['median_stall_s']:.0f} s) account for "
                       f"**{st['clock_share'] * 100:.1f}% of the wall clock**.")
            if st["period"]:
                rate = sustained_loader_rate(st, ta["s_per_step"], BATCH)
                need = BATCH / ta["s_per_step"]
                out.append("")
                out.append(f"They land **exactly every {st['period']} steps** -- a prefetch sawtooth, not slow "
                           f"data. {st['period']} is the pipeline depth (workers x prefetch factor): the "
                           f"trainer drains a full queue at the GPU-bound rate, then blocks while the workers "
                           f"refill it.")
                if rate:
                    out.append("")
                    out.append(f"Sustained loader rate **{rate:.1f} samples/s** against the **{need:.1f} "
                               f"samples/s** a {ta['s_per_step']:.2f} s step consumes -- so roughly "
                               f"**{need / rate:.1f}x** more workers removes the stall and floors "
                               f"step time at {ta['s_per_step']:.2f} s.")
                    for n_steps in (10_000, 30_000):
                        h = n_steps * ta["s_per_step"] / 3600
                        out.append(f"  - if fixed: {n_steps:,} steps ~ {h:.1f} h ~ ${h * 0.72:.2f} per arm")
            else:
                out.append("")
                out.append("They are NOT periodic, so this is not a prefetch artefact -- do not treat adding "
                           "workers as the fix without measuring again.")
    else:
        out.append("not enough steady-state steps logged")

    out += ["", "## Rung 5 -- does steps/s depend on task count?", ""]
    if ta["steps_per_s"] and tm["steps_per_s"]:
        ratio = tm["steps_per_s"] / ta["steps_per_s"]
        out.append(f"one task {ta['steps_per_s']:.3f} steps/s, two tasks {tm['steps_per_s']:.3f} steps/s, "
                   f"ratio {ratio:.2f}. Within ~10% means batch-bound as expected, so k is a CONVERGENCE "
                   f"question that throughput cannot settle.")
    else:
        out.append("not computable from the logs")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    print(report(Path(sys.argv[1])))
