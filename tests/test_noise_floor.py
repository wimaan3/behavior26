"""sigma_w estimation, pinned. Pure Python."""
from __future__ import annotations

import json
import math

import pytest

from analysis.noise_floor import estimate, load_rollouts, report


def _rollouts(pattern):
    """pattern: {instance: [q, q, ...]}"""
    return [{"instance_id": i, "rollout_id": r, "q": q, "success": bool(q)}
            for i, qs in pattern.items() for r, q in enumerate(qs)]


def test_deterministic_instances_give_zero_and_say_so():
    nf = estimate(_rollouts({0: [0.0] * 6, 1: [1.0] * 6}))
    assert nf.sigma_w == 0.0 and nf.n_flipped == 0
    assert "upper bound" in report(nf)


def test_a_coin_flip_instance_gives_the_textbook_variance():
    """p=0.5, r=6: unbiased pi(1-pi) estimate is 6/5 * 0.25 = 0.3, sigma_w = sqrt(0.3)."""
    nf = estimate(_rollouts({0: [0, 1, 0, 1, 0, 1]}))
    assert nf.per_instance_var[0] == pytest.approx(0.3)
    assert nf.sigma_w == pytest.approx(math.sqrt(0.3))


def test_pooling_averages_variances_not_standard_deviations():
    nf = estimate(_rollouts({0: [0, 1, 0, 1, 0, 1], 1: [0.0] * 6}))
    assert nf.sigma_w == pytest.approx(math.sqrt(0.3 / 2))


def test_f_and_the_seed_decision_follow_sigma_w():
    """sigma_w = 0.5*sqrt(f) on a binary task, so f = 4 sigma_w^2."""
    nf = estimate(_rollouts({0: [0, 1, 0, 1, 0, 1]}))
    assert nf.f_implied == pytest.approx(4 * 0.3)
    assert "3 seeds" in report(nf)          # f = 1.2 > 0.2
    quiet = estimate(_rollouts({i: [0.0] * 6 for i in range(11)} | {11: [0, 0, 0, 0, 0, 1]}))
    assert "1 seed" in report(quiet)        # f small


def test_single_rollout_per_instance_is_refused():
    with pytest.raises(ValueError, match="single rollout"):
        estimate(_rollouts({0: [1.0], 1: [0.0]}))


def test_graded_task_uses_the_sample_variance(tmp_path):
    nf = estimate(_rollouts({0: [0.2, 0.4, 0.6, 0.4, 0.2, 0.6]}))
    assert not nf.binary and nf.f_implied is None
    assert nf.sigma_w == pytest.approx(0.17889, abs=1e-4)   # sqrt(sample variance 0.032)


def test_loads_evaluator_json_shape(tmp_path):
    d = tmp_path / "json"; d.mkdir()
    for r in range(2):
        (d / f"turning_on_radio_0_{r}.json").write_text(json.dumps(
            {"task": "turning_on_radio", "instance_id": 0, "rollout_id": r,
             "success": r == 1, "q_score": {"final": float(r)}}))
    nf = estimate(load_rollouts(d))
    assert nf.n_instances == 1 and nf.n_flipped == 1


def test_the_runner_uses_the_baseline_not_the_null_policy():
    """A null policy scores ~0 everywhere: within-instance variance is trivially
    zero and sigma_w would read 0. The protocol is explicit about this."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "scripts" / "session_a" / "noise_floor.sh").read_text()
    assert "serve_baseline.sh" in text and "null_server" not in text
    assert "--num-rollouts \"$REPEATS\"" in text, "repeats must reach the evaluator"
    assert "--instance-indices $(seq" in text, "instances must be spread, not one instance repeated"


def _runner_text():
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "scripts" / "session_a" / "noise_floor.sh").read_text()


def test_the_runner_does_not_write_video_it_never_reads():
    """sigma_w is estimated from the JSON alone. 72 full-res RGBD rollouts of video is
    tens of GB of container disk for nothing, and first_rollout.sh already lost a run
    to exactly this (full-res video into 8 GB of volume)."""
    text = _runner_text()
    assert "--write-video" not in text, "the noise floor reads JSON; video is pure disk risk"


def test_the_runner_evaluates_training_instances():
    """The evaluator's --mode defaults to public_test. sigma_w must be measured on the
    same instance pool the A/B will use."""
    assert "--mode train" in _runner_text()


def test_the_runner_leaves_scene_reuse_on():
    """Scene load is per evaluator invocation. --instances-per-job defaults to 0 (one
    job for everything), which is what makes 72 rollouts cost ~4 h instead of ~24."""
    assert "--instances-per-job" not in _runner_text()


def test_the_runner_checks_it_got_every_rollout():
    """A partial run silently biases sigma_w toward whichever instances finished."""
    text = _runner_text()
    assert "INSTANCES * REPEATS" in text
    assert "-eq \"$((INSTANCES * REPEATS))\"" in text, "must require ALL rollouts, not merely some"
