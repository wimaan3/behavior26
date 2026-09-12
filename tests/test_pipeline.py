"""
Regression tests for the GPU-free pipeline.

Every test in the "bugs found" section below pins a defect that was live in this repo
and that only shows up on data we could not produce before the mock evaluator existed.
They are cheap; run them before every push.

    pytest tests/test_pipeline.py -q        # or: python -m tests.test_pipeline
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.compare import load_pair, paired_stats          # noqa: E402
from analysis.failures import Thresholds, tag_row             # noqa: E402
from analysis.parse import load_rollouts, summarize           # noqa: E402
from harness.launch import Job, already_done, completed_keys  # noqa: E402
from policy.wire import action_length, extract_action, pack, unpack  # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def write_rollout(json_dir: Path, task: str, inst: int, q: float | None = 0.0,
                  rollout: int = 0, steps: int = 100, max_steps: int = 500,
                  base: float = 0.0, left: float = 0.0, right: float = 0.0) -> Path:
    json_dir.mkdir(parents=True, exist_ok=True)
    d: dict = {
        "task": task, "instance_id": inst, "rollout_id": rollout, "steps": steps,
        "success": bool(q is not None and q >= 1.0),
        "agent_distance": {"base": base, "left": left, "right": right},
        "normalized_agent_distance": base / 10.0,
        "time": {"simulator_steps": steps, "simulator_time": steps / 13.5,
                 "normalized_time": steps / max_steps},
    }
    if q is not None:
        d["q_score"] = {"final": q}
    path = json_dir / f"{task}_inst{inst}_rollout{rollout}.json"
    path.write_text(json.dumps(d))
    return path


# --------------------------------------------------------------------------------------
# wire protocol
# --------------------------------------------------------------------------------------
def test_wire_roundtrip():
    msg = {"action": [0.0] * 23, "meta": {"n": 1}}
    assert unpack(pack(msg)) == msg


def test_extract_action_accepts_both_documented_and_openpi_keys():
    assert extract_action({"action": [1, 2, 3]}) == [1, 2, 3]
    assert extract_action({"actions": [1, 2, 3]}) == [1, 2, 3]
    assert extract_action([1, 2, 3]) == [1, 2, 3]


def test_action_length_handles_flat_and_chunked():
    assert action_length([0.0] * 23) == 23
    assert action_length([[0.0] * 23 for _ in range(16)]) == 23


def test_unpack_rejects_text_frames():
    # A text frame means the peer is not speaking msgpack; fail loudly, not silently.
    try:
        unpack("not msgpack")
    except TypeError:
        return
    raise AssertionError("expected TypeError on a text frame")


# --------------------------------------------------------------------------------------
# BUG 1: already_done matched filenames by substring
# --------------------------------------------------------------------------------------
def test_resume_does_not_confuse_instance_1_with_instance_10(tmp_path):
    """'1' is a substring of 'inst10' and '0' of 'rollout0'.

    The old filename-substring check reported instances 0 and 1 as complete when only
    10-19 had run. On a resumed sweep that silently drops rollouts, and missing
    instances score zero on the leaderboard.
    """
    for i in range(10, 20):
        write_rollout(tmp_path / "json", "turning_on_radio", i)
    done = completed_keys(tmp_path)

    # Train-mode jobs: ids and CLI indices coincide. See
    # test_resume_matches_instance_ids_not_indices for the test-mode case, where
    # they differ and used to be compared wrongly.
    def train_job(task, ids):
        return Job(task, list(ids), list(ids))

    assert already_done(train_job("turning_on_radio", range(10, 20)), done) is True
    assert already_done(train_job("turning_on_radio", [1]), done) is False
    assert already_done(train_job("turning_on_radio", [0]), done) is False
    assert already_done(train_job("turning_on_radio", [0, 1, 2]), done) is False
    # and no cross-task contamination via substring
    assert already_done(train_job("radio", [10]), done) is False


def test_resume_ignores_truncated_json_so_the_job_reruns(tmp_path):
    write_rollout(tmp_path / "json", "t", 0)
    (tmp_path / "json" / "t_inst1_rollout0.json").write_text("{ truncated")
    done = completed_keys(tmp_path)
    assert ("t", 0) in done
    assert ("t", 1) not in done


# --------------------------------------------------------------------------------------
# BUG 2 & 3: CRASHED and TIMEOUT were unreachable
# --------------------------------------------------------------------------------------
def test_crashed_rollout_is_tagged_crashed_not_scored_as_zero(tmp_path):
    """A rollout JSON with no q_score field is a crash, not a legitimate zero."""
    write_rollout(tmp_path / "json", "t", 0, q=None)
    row = load_rollouts(tmp_path).iloc[0]
    assert row["q_missing"] is True or bool(row["q_missing"])
    assert tag_row(row, Thresholds()) == "CRASHED"


def test_timeout_is_reachable(tmp_path):
    """Ran the full step budget while moving and manipulating -> TIMEOUT."""
    write_rollout(tmp_path / "json", "t", 0, q=0.0, steps=500, max_steps=500,
                  base=5.0, left=1.0, right=1.0)
    row = load_rollouts(tmp_path).iloc[0]
    assert tag_row(row, Thresholds()) == "TIMEOUT"


def test_active_no_progress_still_reachable(tmp_path):
    write_rollout(tmp_path / "json", "t", 0, q=0.0, steps=100, max_steps=500,
                  base=5.0, left=1.0, right=1.0)
    row = load_rollouts(tmp_path).iloc[0]
    assert tag_row(row, Thresholds()) == "ACTIVE_NO_PROGRESS"


def test_missing_distances_do_not_land_in_active_no_progress():
    """NaN is truthy, so `row.get(x) or 0.0` did NOT default missing distances.

    They fell through to ACTIVE_NO_PROGRESS -- the bucket that says 'the robot moved and
    manipulated and still failed', i.e. the one you would spend GPU time investigating.
    """
    row = pd.Series({"q_score": 0.0, "q_missing": False, "dist_base": float("nan"),
                     "dist_left": float("nan"), "dist_right": float("nan"),
                     "normalized_time": float("nan")})
    assert tag_row(row, Thresholds()) != "ACTIVE_NO_PROGRESS"


def test_solved_and_partial(tmp_path):
    write_rollout(tmp_path / "json", "t", 0, q=1.0)
    write_rollout(tmp_path / "json", "t", 1, q=0.4)
    df = load_rollouts(tmp_path)
    tags = {r["instance_id"]: tag_row(r, Thresholds()) for _, r in df.iterrows()}
    assert tags[0] == "SOLVED"
    assert tags[1] == "PARTIAL"


def test_summarize_reports_missing_q_count(tmp_path):
    write_rollout(tmp_path / "json", "t", 0, q=0.5)
    write_rollout(tmp_path / "json", "t", 1, q=None)
    stats = summarize(load_rollouts(tmp_path))
    assert stats["rollouts_missing_q"] == 1
    assert stats["rollouts"] == 2


# --------------------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------------------
def test_paired_stats_ci_and_significance(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # B is uniformly +0.10 on every instance: an unmistakable, perfectly paired win.
    for i in range(12):
        write_rollout(a / "json", "t", i, q=0.10 + 0.01 * i)
        write_rollout(b / "json", "t", i, q=0.20 + 0.01 * i)
    merged, coverage = load_pair(a, b)
    assert coverage["identical_coverage"]
    assert coverage["n_paired"] == 12

    stats = paired_stats(merged)
    assert abs(stats["mean_dq"] - 0.10) < 1e-6
    assert stats["significant"] is True
    assert stats["ci_low"] <= 0.10 <= stats["ci_high"]


def test_paired_detects_mismatched_coverage(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(5):
        write_rollout(a / "json", "t", i, q=0.1)
    for i in range(4):
        write_rollout(b / "json", "t", i, q=0.2)
    merged, coverage = load_pair(a, b)
    assert coverage["identical_coverage"] is False
    assert coverage["only_in_a"] == [("t", 4)]
    assert coverage["n_paired"] == 4


def test_pairing_unit_is_instance_not_rollout_id(tmp_path):
    """Pairing must join on (task, instance), NOT on rollout_id.

    The evaluator is nondeterministic, so arm A's rollout 3 and arm B's rollout 3
    are two unrelated draws from the same instance. Joining on rollout_id pairs
    them anyway, which does two bad things at once: it fails to cancel the
    within-instance noise it was supposed to cancel, and it counts n*m pairs where
    there are only n independent units, so the t-test gets degrees of freedom it
    has not earned.

    Here the treatment is worth a near-constant +0.020 on every one of 6
    instances. Each arm draws 3 seeds whose noise happens to land in a different
    order -- exactly what nondeterminism means. The effect is unmissable at the
    instance level; the rollout_id join buries it under noise of its own making.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    base = [0.10, 0.30, 0.50, 0.20, 0.40, 0.60]
    effect = [0.018, 0.022, 0.019, 0.021, 0.020, 0.020]
    noise = [-0.30, 0.0, 0.30]
    for i, (bs, eff) in enumerate(zip(base, effect)):
        for r, nz in enumerate(noise):
            write_rollout(a / "json", "t", i, q=bs + nz, rollout=r)
        for r, nz in enumerate(reversed(noise)):
            write_rollout(b / "json", "t", i, q=bs + eff + nz, rollout=r)

    merged, coverage = load_pair(a, b)
    # 6 instances x 3 seeds x 2 arms = 36 rollouts, but only 6 units of analysis.
    assert coverage["n_paired"] == 6
    assert coverage["seeds_per_unit_a"] == 3
    assert coverage["seeds_per_unit_b"] == 3

    stats = paired_stats(merged)
    assert stats["n_pairs"] == 6
    assert abs(stats["mean_dq"] - 0.020) < 1e-6
    # The per-instance effect is near-constant, so the paired SD is tiny and the
    # effect is resolved easily. Joining on rollout_id gave sd 0.504, an MDE of
    # 0.25, and "not significant" on this same data.
    assert stats["sd_dq"] < 0.01
    assert stats["min_detectable_dq"] < 0.01
    assert stats["significant"] is True


def test_pairing_averages_seeds_within_arm(tmp_path):
    """d_i is the difference of per-instance MEANS, matching analysis/power.py.

    power.py sizes the experiment as d_i = mean_m Q_B(i) - mean_m Q_A(i). If
    compare.py computed anything else, the design we bought and the number we
    report would be about different quantities.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    for r, q in enumerate([0.0, 0.30, 0.60]):        # A instance mean = 0.30
        write_rollout(a / "json", "t", 0, q=q, rollout=r)
    for r, q in enumerate([0.40, 0.50, 0.90]):       # B instance mean = 0.60
        write_rollout(b / "json", "t", 0, q=q, rollout=r)

    merged, _ = load_pair(a, b)
    assert len(merged) == 1
    assert abs(float(merged["q_score_a"].iloc[0]) - 0.30) < 1e-9
    assert abs(float(merged["q_score_b"].iloc[0]) - 0.60) < 1e-9
    assert abs(float(merged["dq"].iloc[0]) - 0.30) < 1e-9


def test_single_rollout_runs_are_unaffected(tmp_path):
    """The dev loop runs num_rollouts: 1. The fix must be a no-op there.

    With one rollout per instance the mean over seeds is that rollout, so every
    number this module reported before the fix still holds for m=1 data.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(12):
        write_rollout(a / "json", "t", i, q=0.10 + 0.01 * i)
        write_rollout(b / "json", "t", i, q=0.20 + 0.01 * i)
    merged, coverage = load_pair(a, b)
    assert coverage["n_paired"] == 12
    assert coverage["seeds_per_unit_a"] == 1
    stats = paired_stats(merged)
    assert abs(stats["mean_dq"] - 0.10) < 1e-6
    assert stats["significant"] is True


def test_unequal_seed_counts_are_reported(tmp_path):
    """Unequal m across arms is legal but not free -- say so rather than hide it.

    Var(d_i) = sigma_w^2 (1/m_a + 1/m_b) + sigma_b^2, so units with fewer seeds
    are noisier, and the equal-weight mean over units is no longer efficient.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(4):
        write_rollout(a / "json", "t", i, q=0.20, rollout=0)
        for r in range(3):
            write_rollout(b / "json", "t", i, q=0.30, rollout=r)
    merged, coverage = load_pair(a, b)
    assert coverage["n_paired"] == 4
    assert coverage["seeds_per_unit_a"] == 1
    assert coverage["seeds_per_unit_b"] == 3
    assert coverage["balanced_seeds"] is False


def test_paired_beats_unpaired_when_instances_differ_in_difficulty(tmp_path):
    """The whole justification for pairing, as an assertion.

    Instances differ hugely in difficulty; the treatment effect is a constant +0.05.
    Paired sees it cleanly; unpaired is swamped by the between-instance spread.
    """
    from analysis.compare import unpaired_stats
    a, b = tmp_path / "a", tmp_path / "b"
    difficulty = [0.02, 0.55, 0.11, 0.78, 0.31, 0.05, 0.62, 0.24, 0.90, 0.40]
    for i, d in enumerate(difficulty):
        write_rollout(a / "json", "t", i, q=d)
        write_rollout(b / "json", "t", i, q=min(1.0, d + 0.05))
    merged, _ = load_pair(a, b)
    assert paired_stats(merged)["min_detectable_dq"] < \
        unpaired_stats(merged)["min_detectable_dq"]


# --------------------------------------------------------------------------------------
# live protocol test: null server <-> mock evaluator
# --------------------------------------------------------------------------------------
def _wait_healthy(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def test_null_server_and_mock_evaluator_speak_the_same_protocol(tmp_path):
    port = 8711
    srv = subprocess.Popen(
        [sys.executable, "-m", "policy.null_server", "--port", str(port), "--action-dim", "23"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait_healthy(port), "null server never became healthy"
        rc = subprocess.run(
            [sys.executable, "-m", "tests.mock_evaluator",
             "--task-name", "turning_on_radio", "--port", str(port),
             "--instance-indices", "0", "1", "--output-dir", str(tmp_path),
             "--fast", "--seed", "3"],
            cwd=REPO, capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr

        df = load_rollouts(tmp_path)
        assert len(df) == 2
        # A null policy emits zero actions, so it must record zero displacement and
        # therefore tag IMMOBILE wherever it scored nothing.
        assert (df["dist_base"] == 0).all()
        for _, r in df.iterrows():
            if r["q_score"] == 0:
                assert tag_row(r, Thresholds()) == "IMMOBILE"
    finally:
        srv.terminate()
        srv.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"], cwd=REPO))


# --------------------------------------------------------------- instance ids
# Verified against BEHAVIOR-1K v3.9.2: TEST_INSTANCE_IDS = range(301, 341),
# public = 301-320, hidden = 321-340, anything below 301 is a training instance.
# Configs declare real IDS; harness/launch.py converts to the --instance-indices
# the evaluator wants. The old convention ("dev on 10-19") named scored public
# instances, so these tests pin the split rules rather than just the arithmetic.


def test_public_ids_map_to_split_indices():
    from harness.launch import indices_for_ids

    assert indices_for_ids([301, 302, 303], "public_test") == [0, 1, 2]
    assert indices_for_ids([301, 320], "public_test") == [0, 19]
    assert indices_for_ids([321, 340], "hidden_test") == [0, 19]


def test_train_ids_pass_through():
    from harness.launch import indices_for_ids

    assert indices_for_ids([0, 7, 300], "train") == [0, 7, 300]


def test_train_mode_refuses_test_instances():
    """OUR policy, deliberately stricter than the evaluator.

    Upstream does no validation at all in train mode:

        # evaluator.py :: resolve_instance_ids
        if mode == "train":
            return [int(instance_id) for instance_id in instance_indices]

    We refuse because every public-test instance is scored, so there is no
    holdout inside the public set and a dev loop that lands on 301-320 is
    tuning on the leaderboard -- which is exactly what the old "dev loop on
    instances 10-19" config did. If this test ever fails because someone
    relaxed the guard to match upstream, that is a regression, not a fix.
    """
    import pytest as _pytest

    from harness.launch import indices_for_ids

    with _pytest.raises(ValueError, match="TEST instances"):
        indices_for_ids([301], "train")
    with _pytest.raises(ValueError, match="TEST instances"):
        indices_for_ids([10, 11, 340], "train")


def test_the_two_test_splits_do_not_mix():
    import pytest as _pytest

    from harness.launch import indices_for_ids

    with _pytest.raises(ValueError, match="other test split"):
        indices_for_ids([321], "public_test")
    with _pytest.raises(ValueError, match="other test split"):
        indices_for_ids([301], "hidden_test")


def test_training_ids_in_a_test_mode_are_rejected():
    """Catches a config that kept the old index-style `instances: [0, 1, 2]`."""
    import pytest as _pytest

    from harness.launch import indices_for_ids

    with _pytest.raises(ValueError, match="training instances"):
        indices_for_ids([0, 1, 2], "public_test")


def test_resume_matches_instance_ids_not_indices(tmp_path):
    """Regression: resume compared CLI indices (0,1,2) against ids (301,302,303).

    It matched nothing, so a half-finished sweep silently re-ran every job --
    while still reporting the results it had found on disk, so it looked fine.
    On a 2,000-rollout submission run that is hundreds of rented GPU-hours.
    """
    import json as _json

    from harness.launch import already_done, build_jobs, completed_keys

    cfg = {
        "name": "t",
        "tasks": ["can_meat"],
        "instances": [301, 302, 303],
        "mode": "public_test",
    }
    (job,) = build_jobs(cfg, instances_per_job=0)
    assert job.instances == [301, 302, 303]   # what lands on disk
    assert job.indices == [0, 1, 2]           # what goes on the command line

    json_dir = tmp_path / "json"
    json_dir.mkdir()
    for instance in (301, 302, 303):
        (json_dir / f"can_meat_{instance}_0.json").write_text(
            _json.dumps({"task": "can_meat", "instance_id": instance, "rollout_id": 0})
        )

    done = completed_keys(tmp_path)
    assert already_done(job, done), "resume failed to recognise a completed job"

    (json_dir / "can_meat_302_0.json").unlink()
    assert not already_done(job, completed_keys(tmp_path))


def test_build_command_sends_indices_not_ids():
    """The evaluator asserts indices are in range(20); sending 301 would fail there."""
    from pathlib import Path as _Path

    from harness.launch import build_command, build_jobs

    cfg = {"name": "t", "tasks": ["can_meat"], "instances": [301, 302], "mode": "public_test"}
    (job,) = build_jobs(cfg, instances_per_job=0)
    cmd = build_command(job, cfg, port=8000, output_dir=_Path("/tmp/out"))

    assert cmd[cmd.index("--mode") + 1] == "public_test"
    i = cmd.index("--instance-indices")
    assert cmd[i + 1 : i + 3] == ["0", "1"], cmd[i : i + 3]


def test_load_config_rejects_a_dev_loop_pointed_at_scored_instances(tmp_path):
    """A bad config must fail at load, not after a scene load on a rented card."""
    import pytest as _pytest
    import yaml as _yaml

    from harness.launch import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        _yaml.safe_dump({"name": "bad", "tasks": ["can_meat"], "instances": [311], "mode": "train"})
    )
    with _pytest.raises(ValueError, match="TEST instances"):
        load_config(path)
