"""Shot one -- the run the whole project pays for.

Two arms, 10k-30k steps each, 20-70 h apiece at rung 4's measured rate. Every
defect here costs a run that cannot be casually repeated inside a $200 budget, and
the dangerous ones are the quiet kind: arms that differ in something other than the
treatment still produce a ΔQ, and it reads like a result.

These tests are static reads of the runner. They cannot prove it trains; they pin
the properties that make its output mean what it claims.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOT = REPO / "scripts" / "shot_one.sh"


def text() -> str:
    return SHOT.read_text()


def test_the_runner_exists_and_is_executable_bash():
    assert SHOT.exists(), "scripts/shot_one.sh"
    assert text().startswith("#!/usr/bin/env bash")


# --- the arms must differ ONLY in the treatment ------------------------------

def test_there_is_exactly_one_trainer_invocation_shared_by_both_arms():
    """The strongest available guarantee that the arms match: not two invocations
    that happen to agree, but ONE, with the treatment passed in as extra arguments.
    Two literal blocks can drift apart in a later edit; this cannot.
    """
    t = text()
    assert t.count("train_b1k_rooted.py") == 1, (
        "two invocations can drift; keep one shared launcher call")
    assert "run_arm armA" in t and "run_arm armB" in t


def test_the_shared_invocation_fixes_steps_root_assets_and_protocol():
    """Everything that must be identical is spelled once, inside the shared call."""
    t = text()
    body = t[t.index("run_arm ()"):t.index("stage 2_arm_A")]
    for required in ("--num-train-steps $STEPS", "--dataset-root $ROOT",
                     "--assets-base-dir $ASSETS", "--protocol"):
        assert required in body, required


def test_the_only_per_arm_difference_is_the_treatment():
    """Arm A is called with no extra arguments at all, so there is nowhere for an
    accidental difference to live."""
    t = text()
    assert re.search(r"run_arm armA \|\|", t), "arm A takes no extra arguments"
    armb = [l for l in t.splitlines() if l.startswith("run_arm armB")][0]
    assert "--progress-loss-weight" in armb
    assert armb.count("--") == 1, f"arm B carries the head and nothing else: {armb}"


def test_only_arm_b_carries_the_progress_head():
    t = text()
    assert t.count("--progress-loss-weight") == 1, "the treatment belongs to arm B alone"
    assert "$LAMBDA" in t


def test_lambda_is_a_calibrated_input_not_a_literal():
    """Rung 3 measured it (~0.14-0.15 for a 20% gradient share). A number hard-coded
    here would silently outlive the calibration it came from."""
    t = text()
    assert re.search(r'LAMBDA="\$\{LAMBDA:-', t), "LAMBDA must be an overridable parameter"
    assert "rung 3" in t.lower() or "calibrat" in t.lower()


# --- it must survive the pod -------------------------------------------------

def test_training_can_resume_and_checkpoints_often_enough_to_be_worth_resuming():
    t = text()
    assert "--resume" in t
    assert "--save-interval $SAVE_EVERY" in t
    assert re.search(r'SAVE_EVERY="\$\{SAVE_EVERY:-(\d+)', t)
    m = re.search(r'SAVE_EVERY="\$\{SAVE_EVERY:-(\d+)', t)
    assert int(m.group(1)) <= 2000, "a save interval near the run length leaves nothing to resume from"


def test_resume_is_not_paired_with_overwrite():
    """--overwrite discards exactly what --resume exists to continue from."""
    t = text()
    for line in t.splitlines():
        if "--resume" in line:
            assert "--overwrite" not in line


def test_an_arm_that_already_finished_is_not_retrained():
    """Re-invocation after an interruption must cost the remaining work, not all of
    it. At 68 h per arm, redoing a finished arm is a week of budget."""
    t = text()
    assert "arm_done" in t or "ALREADY_DONE" in t


# --- the box is rented -------------------------------------------------------

def test_no_credential_is_written_to_the_box():
    """Standing rule: no private key, token or credential is copied onto rented
    hardware. A push target is fine; a token in a file is not."""
    t = text()
    assert "huggingface-cli login" not in t
    assert not re.search(r"hf_[A-Za-z0-9]{20}", t), "no literal token"
    assert not re.search(r"(>|>>)\s*\S*token", t), "must not write a token to any file"


def test_pushing_requires_a_token_from_the_environment_and_says_so_early():
    """Discovering the token is missing after 68 h of training is the wrong time."""
    t = text()
    assert "HF_TOKEN" in t
    assert "PUSH" in t


# --- the result must be interpretable a month later --------------------------

def test_it_records_a_manifest_of_what_was_actually_run():
    """Steps, lambda, seed, data fingerprint, and the openpi commit. Without these a
    checkpoint is an artifact nobody can attribute."""
    t = text()
    assert "manifest" in t.lower()
    for key in ("lambda", "steps", "seed"):
        assert key in t.lower(), key


def test_norm_stats_are_computed_once_and_shared_by_both_arms():
    """Separate stats per arm is a difference between the arms that is not the
    treatment. Rung 1 proved the shared features come out byte-identical; this keeps
    it true by construction rather than by luck."""
    t = text()
    assert t.count("compute_norm_stats_b1k.py") == 1, "one stats run, shared"


def test_training_failure_is_not_swallowed_by_a_pipe():
    """The trainer's output goes through a timestamping pipe, so a naive gate reports
    the TIMESTAMPER's status and records a failed 68-hour run as a success.

    pipefail makes the subshell's own status non-zero when the trainer fails, so
    `rc=$?` on the subshell is correct. PIPESTATUS[0] would also work here but reads
    as if it indexes the inner pipe, which it does not -- after a subshell the array
    holds one element, the subshell's status. Require the honest form.
    """
    t = text()
    assert "pipefail" in t, "without pipefail the pipe hides the trainer's exit code"
    assert re.search(r"^\s*local rc=\$\?", t, re.M), "capture the subshell's own status"
    code = [l for l in t.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert not any("PIPESTATUS" in l for l in code), (
        "PIPESTATUS after a subshell reads like the inner pipe but is not; use $?")


def test_the_recorded_seed_is_the_seed_actually_used():
    """A manifest that reports a seed the trainer never received is a reproduction
    instruction that does not reproduce."""
    t = text()
    assert "--seed $SEED" in t, "SEED must reach the trainer, not only the manifest"
