"""The rung report's rules, pinned against synthetic logs. Pure Python."""
from __future__ import annotations

import math
import random

from analysis.rung_report import branch0, paired_action, parse, throughput


def _log(n, *, progress=None, action=None, dt=0.5, t0=1000.0):
    lines = []
    for s in range(n):
        parts = [f"action_loss={action(s):.4f}" if action else "action_loss=0.8"]
        if progress is not None:
            parts.append(f"progress_loss={progress(s)}")
        lines.append(f"{t0 + s * dt:.3f} Step {s}: " + ", ".join(parts))
    return "\n".join(lines)


def test_parse_reads_timestamped_step_lines():
    run = parse(_log(3, progress=lambda s: 0.69))
    assert run.steps == [0, 1, 2] and run.times[1] == 1000.5
    assert run.series("progress_loss") == [0.69, 0.69, 0.69]


def test_absent_progress_loss_is_a_broken_run():
    assert branch0(parse(_log(300)))["verdict"] == "BROKEN"


def test_nan_is_a_broken_run():
    assert branch0(parse(_log(300, progress=lambda s: "nan" if s == 150 else 0.69)))["verdict"] == "BROKEN"


def test_noisy_but_flat_is_broken():
    rng = random.Random(0)
    run = parse(_log(1000, progress=lambda s: round(0.693 + rng.gauss(0, 0.01), 4)))
    assert branch0(run)["verdict"] == "BROKEN"


def test_a_real_decrease_is_learning():
    rng = random.Random(1)
    run = parse(_log(1000, progress=lambda s: round(0.693 - 0.2 * s / 1000 + rng.gauss(0, 0.01), 4)))
    r = branch0(run)
    assert r["verdict"] == "LEARNING" and r["z"] > 2


def test_too_short_is_inconclusive_not_a_verdict():
    assert branch0(parse(_log(150, progress=lambda s: 0.69)))["verdict"] == "INCONCLUSIVE"


def test_identical_action_losses_are_not_degraded():
    a = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s)))
    b = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s), progress=lambda s: 0.69))
    assert paired_action(a, b)["verdict"] == "NOT_DEGRADED"


def test_consistently_higher_action_loss_in_B_is_degraded():
    a = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s)))
    b = parse(_log(300, action=lambda s: 0.85 + 0.1 * math.sin(s), progress=lambda s: 0.69))
    assert paired_action(a, b)["verdict"] == "DEGRADED"


def test_throughput_excludes_compile_steps():
    """Step 0-49 slow (compile), then 0.5 s/step."""
    lines = []
    t = 0.0
    for s in range(300):
        t += 20.0 if s < 50 else 0.5
        lines.append(f"{t:.3f} Step {s}: action_loss=0.8")
    r = throughput(parse("\n".join(lines)))
    assert abs(r["steps_per_s"] - 2.0) < 1e-6
