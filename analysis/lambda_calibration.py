"""Choose progress_loss_weight (lambda) from measured gradient norms.

Rung 3 of session B. `progress_loss_weight = 0.1` was set before anyone had seen
either loss; the protocol replaces it with a calibrated value.

WHAT IS TARGETED
----------------
The share of the combined gradient magnitude carried by the progress term:

    share(lambda) = lambda*|g_p| / (|g_a| + lambda*|g_p|)

where g_a = grad(action_loss) and g_p = grad(progress_loss) UNweighted, both over
the trainable parameters, as logged by train_step when
`log_loss_term_grad_norms=True` (`grad_norm_action`, `grad_norm_progress_raw`).
Target: a meaningful minority, 10-30%. Solving share(lambda) = s:

    lambda* = s/(1-s) * |g_a| / |g_p|

WHY GRADIENT AND NOT LOSS
-------------------------
A loss share says nothing about the update. Measured at rung 1, step 0: the
progress term was 8% of the loss at lambda=0.1 (0.0708 of 0.8825), but the two
losses have different gradient magnitudes per unit loss, so that 8% can be a much
larger or smaller share of what actually moves the weights.

WHY A MEDIAN OVER STEPS
-----------------------
Per-step gradient norms are noisy (rung 1's grad_norm ranged 1.40-2.32 across ten
steps), and the first steps sit inside LR warmup where the head has not moved.
The ratio |g_a|/|g_p| is taken per step and the median over a window is used, so
a single outlier batch cannot set lambda.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    target_share: float
    ratio_median: float           # median over steps of |g_a| / |g_p|
    lam: float                    # lambda that hits target_share at the median ratio
    share_at_current: float | None
    current_lambda: float | None
    steps_used: int
    cosine_median: float | None   # median cosine(g_a, g_p); < 0 means the head fights actions


def share(lam: float, g_action: float, g_progress_raw: float) -> float:
    """Share of combined gradient magnitude carried by lam * progress."""
    if lam < 0 or g_action < 0 or g_progress_raw < 0:
        raise ValueError("norms and lambda must be non-negative")
    num = lam * g_progress_raw
    den = g_action + num
    return 0.0 if den == 0 else num / den


def lambda_for_share(target: float, g_action: float, g_progress_raw: float) -> float:
    if not 0 < target < 1:
        raise ValueError(f"target share must be in (0, 1), got {target}")
    if g_progress_raw <= 0:
        raise ValueError("progress gradient is zero -- the head is not connected; no lambda can help")
    return target / (1 - target) * g_action / g_progress_raw


_STEP = re.compile(r"Step (\d+): (.*)")


def parse_log(text: str) -> list[dict[str, float]]:
    """`Step N: k=v, k=v` lines from train_b1k.py into dicts."""
    rows = []
    for raw in text.split("\n"):
        # see analysis/rung_report.parse: tqdm's carriage returns mean the last
        # \r-separated segment is the live one.
        line = raw.split("\r")[-1]
        m = _STEP.search(line)
        if not m:
            continue
        row = {"step": float(m.group(1))}
        for part in m.group(2).split(","):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        rows.append(row)
    return rows


def calibrate(rows: list[dict[str, float]], *, target_share: float = 0.2,
              current_lambda: float | None = None, first_step: int = 0) -> Calibration:
    used = [r for r in rows if r["step"] >= first_step
            and "grad_norm_action" in r and "grad_norm_progress_raw" in r
            and r["grad_norm_progress_raw"] > 0 and math.isfinite(r["grad_norm_action"])]
    if not used:
        raise ValueError("no steps with grad_norm_action and grad_norm_progress_raw -- "
                         "was the run launched with log_loss_term_grad_norms=True?")
    ratio = statistics.median(r["grad_norm_action"] / r["grad_norm_progress_raw"] for r in used)
    lam = target_share / (1 - target_share) * ratio
    cur = None
    if current_lambda is not None:
        cur = current_lambda / (ratio + current_lambda)   # share() at the median ratio, |g_p| := 1
    cos = [r["grad_cosine_action_progress"] for r in used if "grad_cosine_action_progress" in r]
    return Calibration(target_share, ratio, lam, cur, current_lambda, len(used),
                       statistics.median(cos) if cos else None)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--target-share", type=float, default=0.2)
    ap.add_argument("--current-lambda", type=float, default=None)
    ap.add_argument("--first-step", type=int, default=0)
    a = ap.parse_args(argv)
    c = calibrate(parse_log(open(a.log).read()), target_share=a.target_share,
                  current_lambda=a.current_lambda, first_step=a.first_step)
    print(f"steps used                 {c.steps_used}")
    print(f"median |g_action|/|g_prog|  {c.ratio_median:.4f}")
    if c.current_lambda is not None:
        print(f"share at lambda={c.current_lambda:<8g}  {c.share_at_current:.1%}")
    for s in (0.10, 0.20, 0.30):
        print(f"lambda for {s:.0%} share         {s / (1 - s) * c.ratio_median:.4f}")
    if c.cosine_median is not None:
        print(f"median cos(g_action,g_prog) {c.cosine_median:+.3f}"
              + ("   <- the head pulls AGAINST the action loss" if c.cosine_median < 0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
