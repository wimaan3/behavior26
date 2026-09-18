"""The go/no-go that decides whether shot one keeps spending.

Pre-registered before any shot-one step runs: over steps WARM..GATE, the MEAN
seconds per step must be within 15% of the GPU-bound 3.90 s rung 4 measured.
The mean, because the mean is what the clock charges; rung 4's median (3.90)
looked fine while the mean (8.11) was more than double.
"""
from __future__ import annotations

from analysis.stall_gate import decide


def _log(n_steps, fast=3.9, stall=0.0, period=24, every=1, t0=1_000_000.0):
    lines, t = [], t0
    for s in range(n_steps + 1):
        if s % every == 0:
            lines.append(f"{t:.1f} \r\rStep {s}: action_loss=1.0")
        t += fast + (stall if stall and (s + 1) % period == 0 else 0.0)
    return "\n".join(lines)


def test_the_rung4_sawtooth_is_a_no_go():
    g = decide(_log(320, stall=103.0), gate_step=300)
    assert g.verdict == "NOGO", g
    assert g.mean_s_per_step > 7


def test_a_flat_gpu_bound_run_is_a_go():
    g = decide(_log(320), gate_step=300)
    assert g.verdict == "GO", g
    assert abs(g.mean_s_per_step - 3.9) < 0.05


def test_too_few_steps_is_wait_not_a_verdict():
    """Deciding on the warm-up would judge compilation, not the loader."""
    g = decide(_log(120), gate_step=300)
    assert g.verdict == "WAIT"


def test_it_works_at_a_coarse_log_interval():
    """A 50-step log interval averages each stall into its window; the MEAN is
    unaffected by that, which is why the gate uses it rather than per-step stalls."""
    assert decide(_log(320, stall=103.0, every=50), gate_step=300).verdict == "NOGO"
    assert decide(_log(320, every=50), gate_step=300).verdict == "GO"


def test_a_mild_slowdown_inside_the_margin_is_still_a_go():
    g = decide(_log(320, fast=4.3), gate_step=300)     # +10%, inside the 15% margin
    assert g.verdict == "GO"
