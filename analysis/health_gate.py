"""Is shot one training healthy? Run by the pod's supervisor every few minutes.

The step-300 stall gate only checks SPEED. A run can be fast and broken: a NaN, a loss
that never falls, or (arm B) a progress head receiving no signal. Each of those would
otherwise burn the rest of a 10-30 h arm before anyone looked.

Rules fixed before shot one, calibrated on the real rung logs (healthy runs: action-loss
drop z ~= 13 and Branch 0 z = 18.7 by step 1000, against a threshold of 2):

    any logged loss is NaN/inf                                        -> FAIL (any time)
    at step CHECK: action_loss over [CHECK-100, CHECK) below [50, 150)
                   by more than 2 standard errors                     -> else FAIL
    arm B at CHECK: Branch 0 on progress_loss (rung_report.branch0),
                   the manipulation check pre-registered 2026-09-15   -> else FAIL
    otherwise, before CHECK                                           -> WAIT
    otherwise                                                         -> PASS

    python -m analysis.health_gate <train log> --arm A|B [--check-step 1000]
exits 0 PASS, 1 FAIL, 3 WAIT.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from analysis.rung_report import Run, _mean_se, branch0, by_step, parse, read_log

CHECK = 1000
EARLY = (50, 150)
WINDOW = 100
Z_MIN = 2.0
LOSS_KEYS = ("loss", "action_loss", "progress_loss")


@dataclass(frozen=True)
class Health:
    verdict: str
    reason: str


def _windowed(run: Run, key: str, check: int) -> Run:
    """The run truncated to steps < check, one value per step (a resume re-logs steps)."""
    vals = {s: v for s, v in by_step(run, key).items() if s < check}
    out = Run()
    for s in sorted(vals):
        out.steps.append(s)
        out.times.append(0.0)
        out.metrics.append({key: vals[s]})
    return out


def decide(text: str, arm: str, check: int = CHECK) -> Health:
    run = parse(text)
    for s, m in zip(run.steps, run.metrics):
        for k in LOSS_KEYS:
            if k in m and not math.isfinite(m[k]):
                return Health("FAIL", f"{k} is NaN/inf at step {s}")
    last = max(run.steps) if run.steps else 0
    if last < check - 1:          # the window is [check-100, check): complete once step check-1 is logged
        return Health("WAIT", f"at step {last}; checks run once step {check - 1} is logged")

    a = by_step(run, "action_loss")
    early = [v for s, v in a.items() if EARLY[0] <= s < EARLY[1]]
    late = [v for s, v in a.items() if check - WINDOW <= s < check]
    if len(early) < 20 or len(late) < 20:
        return Health("FAIL", f"action_loss logged too sparsely to check ({len(early)}, {len(late)} points)")
    m0, s0 = _mean_se(early)
    m1, s1 = _mean_se(late)
    se = math.hypot(s0, s1)
    z = (m0 - m1) / se if se > 0 else float("inf")
    if z <= Z_MIN:
        return Health("FAIL", f"action_loss not falling: steps {EARLY[0]}-{EARLY[1]} {m0:.4f} vs "
                              f"{check - WINDOW}-{check} {m1:.4f}, z = {z:.1f} <= {Z_MIN}")
    reason = f"action_loss {m0:.4f} -> {m1:.4f} (z = {z:.1f})"

    if arm == "B":
        b = branch0(_windowed(run, "progress_loss", check), window=WINDOW)
        if b["verdict"] != "LEARNING":
            return Health("FAIL", f"progress head: Branch 0 {b['verdict']} -- {b['reason']}")
        reason += f"; progress_loss {b['first_mean']:.4f} -> {b['last_mean']:.4f} (z = {b['z']:.1f})"
    return Health("PASS", reason)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--arm", choices=("A", "B"), required=True)
    ap.add_argument("--check-step", type=int, default=CHECK)
    a = ap.parse_args()
    h = decide(read_log(a.log), arm=a.arm, check=a.check_step)
    print(f"HEALTH {h.verdict} arm {a.arm}: {h.reason}")
    return {"PASS": 0, "FAIL": 1, "WAIT": 3}[h.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
