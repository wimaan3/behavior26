"""Is the training run healthy? Checked inside the pod, so it does not depend on anyone watching.

Rules fixed before shot one (calibrated on the real rung-4 and rung-2/3 logs, where a
healthy run shows z ~= 13 for the action-loss drop and z = 18.7 for Branch 0 by step 1000):

    any logged loss NaN/inf                                         -> FAIL, any time
    at step CHECK (1000): action_loss steps 900-999 below 50-149 by >2 SE   else FAIL
    arm B at CHECK: Branch 0 -- progress_loss falling by >2 SE      else FAIL
    before CHECK, nothing wrong yet                                 -> WAIT
"""
from __future__ import annotations

from pathlib import Path

from analysis.health_gate import decide
from analysis.rung_report import read_log

REAL = Path(__file__).resolve().parents[1] / "docs" / "sessionB-2026-09-16-rung2-5" / "logs"


def _log(n, action=lambda s: 0.8 * 0.998 ** s, progress=None, t0=1_000_000.0):
    lines = []
    for s in range(n + 1):
        kv = f"action_loss={action(s):.4f}"
        if progress is not None:
            kv += f", progress_loss={progress(s):.4f}"
        lines.append(f"{t0 + 3.9 * s:.1f} \r\rStep {s}: {kv}")
    return "\n".join(lines)


def _noisy(base, amp=0.15):
    # deterministic jitter so the SE test sees realistic noise
    return lambda s: base(s) + amp * (((s * 7919) % 101) / 101 - 0.5)


def test_the_real_healthy_arm_a_passes():
    assert decide(read_log(REAL / "train_rung4_armA.log"), arm="A").verdict == "PASS"


def test_the_real_healthy_arm_b_passes():
    assert decide(read_log(REAL / "train_rung23_armB.log"), arm="B").verdict == "PASS"


def test_nan_fails_immediately_even_before_the_check_step():
    text = _log(40).replace("Step 30: action_loss=", "Step 30: action_loss=nan, x=")
    g = decide(text, arm="A")
    assert g.verdict == "FAIL" and "nan" in g.reason.lower()


def test_a_flat_action_loss_fails_at_the_check_step():
    text = _log(1000, action=_noisy(lambda s: 0.7))
    g = decide(text, arm="A")
    assert g.verdict == "FAIL" and "action" in g.reason


def test_before_the_check_step_a_healthy_run_waits():
    assert decide(_log(600, action=_noisy(lambda s: 0.8 * 0.998 ** s)), arm="A").verdict == "WAIT"


def test_arm_b_with_no_progress_loss_fails():
    """The head built but not supervised: arm B would silently equal arm A."""
    g = decide(_log(1000, action=_noisy(lambda s: 0.8 * 0.998 ** s)), arm="B")
    assert g.verdict == "FAIL" and "progress" in g.reason


def test_arm_b_with_flat_progress_loss_fails():
    g = decide(_log(1000, action=_noisy(lambda s: 0.8 * 0.998 ** s),
                    progress=_noisy(lambda s: 0.693, amp=0.02)), arm="B")
    assert g.verdict == "FAIL" and "progress" in g.reason


def test_arm_b_learning_passes():
    g = decide(_log(1000, action=_noisy(lambda s: 0.8 * 0.998 ** s),
                    progress=_noisy(lambda s: 0.693 - 0.00005 * s, amp=0.02)), arm="B")
    assert g.verdict == "PASS", g


def test_duplicate_steps_after_a_resume_use_the_latest_value():
    """A resumed run re-logs steps from its checkpoint; the check must not double-count."""
    healthy = _log(1000, action=_noisy(lambda s: 0.8 * 0.998 ** s))
    replay = "\n".join(l for l in healthy.splitlines() if " Step 9" in l and len(l.split("Step ")[1].split(":")[0]) == 3)
    assert decide(healthy + "\n" + replay, arm="A").verdict == "PASS"


# --- warm start (shot two, 8 Oct): both arms start from the released radio checkpoint ------
# A converged start has no falling loss to show, so "must fall" would stop a healthy run.
# What a warm start must show instead is that the checkpoint really loaded and matches the
# data pipeline: the opening loss is far below a from-base start (0.94 over shot one's first
# 20 steps), and it stays there.

def test_a_warm_start_with_a_low_flat_loss_passes():
    g = decide(_log(1000, action=_noisy(lambda s: 0.08, amp=0.04)), arm="A", warm=True)
    assert g.verdict == "PASS", g.reason


def test_a_warm_start_that_opens_at_base_level_fails_within_the_first_steps():
    """Wrong norm stats, or weights that did not load: caught ~20 steps in, not at 1000."""
    g = decide(_log(25, action=_noisy(lambda s: 0.9)), arm="A", warm=True)
    assert g.verdict == "FAIL" and "warm start" in g.reason


def test_a_warm_start_waits_until_the_opening_window_is_logged():
    assert decide(_log(10, action=lambda s: 0.08), arm="A", warm=True).verdict == "WAIT"


def test_a_warm_start_whose_loss_climbs_to_base_level_fails_at_the_check_step():
    g = decide(_log(1000, action=_noisy(lambda s: 0.08 + 0.0006 * s, amp=0.04)), arm="A", warm=True)
    assert g.verdict == "FAIL"


def test_a_warm_arm_b_still_needs_its_progress_head_to_learn():
    flat = _log(1000, action=_noisy(lambda s: 0.08, amp=0.04), progress=_noisy(lambda s: 0.69, amp=0.02))
    assert decide(flat, arm="B", warm=True).verdict == "FAIL"
    learning = _log(1000, action=_noisy(lambda s: 0.08, amp=0.04),
                    progress=_noisy(lambda s: 0.69 * 0.998 ** s, amp=0.02))
    assert decide(learning, arm="B", warm=True).verdict == "PASS"


def test_without_the_flag_the_cold_start_rule_is_unchanged():
    assert decide(_log(1000, action=_noisy(lambda s: 0.08, amp=0.04)), arm="A").verdict == "FAIL"
