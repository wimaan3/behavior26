"""lambda calibration arithmetic, pinned. Pure Python -- runs without JAX."""
from __future__ import annotations

import math

import pytest

from analysis.lambda_calibration import calibrate, lambda_for_share, parse_log, share


def test_lambda_for_share_inverts_share():
    for target in (0.1, 0.2, 0.3):
        lam = lambda_for_share(target, g_action=2.0, g_progress_raw=0.5)
        assert math.isclose(share(lam, 2.0, 0.5), target, rel_tol=1e-12)


def test_equal_norms_give_the_textbook_answer():
    # |g_a| == |g_p|: a 20% share needs lambda = 0.25
    assert math.isclose(lambda_for_share(0.2, 1.0, 1.0), 0.25)


def test_a_disconnected_head_is_an_error_not_a_huge_lambda():
    with pytest.raises(ValueError, match="not connected"):
        lambda_for_share(0.2, 1.0, 0.0)


LOG = """
Step 0: action_loss=0.8116, grad_norm=2.2390, loss=0.8825, progress_loss=0.7084, grad_norm_action=2.0, grad_norm_progress_raw=4.0, grad_cosine_action_progress=-0.10
Step 1: action_loss=1.0469, grad_norm=2.3051, loss=1.1158, progress_loss=0.6886, grad_norm_action=3.0, grad_norm_progress_raw=4.0, grad_cosine_action_progress=0.05
Step 2: action_loss=0.6927, grad_norm=1.5414, loss=0.7618, progress_loss=0.6914, grad_norm_action=90.0, grad_norm_progress_raw=1.0, grad_cosine_action_progress=0.02
"""


def test_parse_reads_train_b1k_step_lines():
    rows = parse_log(LOG)
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[0]["progress_loss"] == pytest.approx(0.7084)


def test_median_resists_one_outlier_step():
    """Step 2's ratio is 90; a mean would drag lambda far off. Median of
    {0.5, 0.75, 90} is 0.75."""
    c = calibrate(parse_log(LOG), target_share=0.2, current_lambda=0.1)
    assert c.ratio_median == pytest.approx(0.75)
    assert c.lam == pytest.approx(0.25 * 0.75)
    assert c.steps_used == 3
    assert c.cosine_median == pytest.approx(0.02)


def test_first_step_skips_warmup_steps():
    c = calibrate(parse_log(LOG), first_step=1)
    assert c.steps_used == 2


def test_a_run_without_term_gradients_says_so():
    rung1_style = "Step 0: action_loss=0.8116, grad_norm=2.2390, loss=0.8825, progress_loss=0.7084"
    with pytest.raises(ValueError, match="log_loss_term_grad_norms"):
        calibrate(parse_log(rung1_style))
