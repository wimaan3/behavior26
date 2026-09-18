"""Go/no-go for shot one at step GATE: is the loader keeping up?

Rung 4 (8 workers) ran at a median 3.90 s/step but a MEAN 8.11 s/step, because a
prefetch stall every 24 steps took 52.8% of the clock. The fix -- more workers --
is a prediction, not a measurement, so shot one measures it on its own opening steps
instead of a separate benchmark, and this decides whether to keep spending.

The rule, fixed before any shot-one step runs:

    over steps WARM..GATE, mean seconds per step <= MAX_MEAN_S   -> GO
    otherwise                                                    -> NOGO
    fewer than GATE steps logged                                 -> WAIT

MAX_MEAN_S = 4.5 is the GPU-bound 3.90 s plus 15%. It is the MEAN because the mean is
what the clock charges; rung 4's median looked fine at double the real cost. The mean
also survives a coarse log interval, which averages each stall into its window.

    python -m analysis.stall_gate <train log> [--gate-step 300] [--max-mean-s 4.5]
exits 0 GO, 1 NOGO, 3 WAIT.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from analysis.rung_report import parse, read_log

WARM = 50
GATE = 300
MAX_MEAN_S = 4.5


@dataclass(frozen=True)
class Gate:
    verdict: str
    mean_s_per_step: float | None
    steps_seen: int
    reason: str


def decide(text: str, gate_step: int = GATE, warm: int = WARM, max_mean_s: float = MAX_MEAN_S) -> Gate:
    run = parse(text)
    pts = [(s, t) for s, t in zip(run.steps, run.times) if warm <= s <= gate_step]
    last = max(run.steps) if run.steps else 0
    if last < gate_step or len(pts) < 2 or pts[-1][0] - pts[0][0] < (gate_step - warm) // 2:
        return Gate("WAIT", None, last, f"at step {last}; the gate reads steps {warm}..{gate_step}")
    (s0, t0), (s1, t1) = pts[0], pts[-1]
    mean = (t1 - t0) / (s1 - s0)
    ok = mean <= max_mean_s
    return Gate("GO" if ok else "NOGO", mean, last,
                f"mean {mean:.2f} s/step over steps {s0}..{s1} "
                f"{'<=' if ok else '>'} {max_mean_s:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--gate-step", type=int, default=GATE)
    ap.add_argument("--max-mean-s", type=float, default=MAX_MEAN_S)
    a = ap.parse_args()
    g = decide(read_log(a.log), gate_step=a.gate_step, max_mean_s=a.max_mean_s)
    print(f"GATE {g.verdict} {g.reason}")
    return {"GO": 0, "NOGO": 1, "WAIT": 3}[g.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
