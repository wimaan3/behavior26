"""Shot one -- the run the whole project pays for.

Two arms, 10k-30k steps each, 20-70 h apiece. Every defect here costs a run that
cannot be casually repeated inside a $200 budget, and the dangerous ones are quiet:
arms that differ in something other than the treatment still produce a ΔQ.

These are static reads of the runner; they cannot prove it trains. The first version
of this file pinned a BUG: it required arm B to carry exactly one flag
(--progress-loss-weight) on arm A's config, which has no progress head -- the patch's
own guard would have raised when arm B started, ~32 h into the run. So these tests
also check the runner against the scripts and configs it calls, not only its shape.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOT = REPO / "scripts" / "shot_one.sh"
PATCH = REPO / "training" / "patches" / "0002-progress-head-and-low-memory-configs.patch"


def text() -> str:
    return SHOT.read_text()


def code_lines() -> list[str]:
    return [l for l in text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def test_the_runner_exists_and_is_bash():
    assert text().startswith("#!/usr/bin/env bash")


# --- the arms must differ ONLY in the treatment ------------------------------

def test_there_is_exactly_one_trainer_invocation():
    """Not two calls that agree today, but one, so they cannot drift apart."""
    assert text().count("scripts/train_b1k_rooted.py") == 1


def test_the_shared_call_fixes_everything_that_must_match():
    t = text()
    body = t[t.index("trainer () {"):t.index("stage 3_check_both_arms")]
    for required in ("--num-train-steps $STEPS", "--dataset-root $ROOT", "--assets-base-dir $ASSETS",
                     "--protocol", "--seed $SEED", "--batch-size", "--num-workers"):
        assert required in body, required


def _arm(name: str) -> list[str]:
    m = re.search(rf"^{name}=\( (.*) \)$", text(), re.M)
    assert m, f"{name} must be declared as an array"
    return shlex.split(m.group(1))


def test_arm_a_is_its_config_and_nothing_else():
    assert _arm("ARM_A") == ["$CFG_A"]


def test_arm_b_is_the_progress_config_plus_the_label_and_the_weight():
    """The head needs its config (progress_head=True), the label column, AND the
    weight. Missing the config: the patch raises when arm B starts. Missing the label:
    the head trains on nothing and arm B silently equals arm A."""
    assert _arm("ARM_B") == ["$CFG_B", "--progress-key", "progress", "--progress-loss-weight", "$LAMBDA"]


def test_the_arm_configs_are_the_pair_the_patch_defines_as_a_clean_ab():
    t, patch = text(), PATCH.read_text()
    assert re.search(r"^CFG_A=pi05_b1k_frozen_vlm$", t, re.M)
    assert re.search(r"^CFG_B=pi05_b1k_frozen_vlm_progress$", t, re.M)
    b = patch[patch.index('name="pi05_b1k_frozen_vlm_progress"'):]
    b = b[:b.index("TrainConfig(") if "TrainConfig(" in b else len(b)]
    assert "progress_head=True" in b, "arm B's config must actually build the head"


def test_both_arms_are_proven_before_either_trains():
    t = text()
    check = t.index("stage 3_check_both_arms")
    first_train = t.index("stage 4_arm_A")
    assert check < first_train
    between = t[check:first_train]
    assert 'trainer armA "${ARM_A[@]}" --check-only' in between
    assert 'trainer armB "${ARM_B[@]}" --check-only' in between
    assert between.count("CONFIG_OK") == 2


def test_norm_stats_are_computed_once_and_the_same_file_goes_to_both_arms():
    t = text()
    assert t.count("compute_norm_stats_b1k.py") == 1
    assert "the two arms' norm stats differ" in t, "must verify, not assume"


def test_the_recorded_seed_is_the_seed_used():
    assert "--seed $SEED" in text()


# --- every flag passed to a script must exist in that script -------------------

def _flags_passed_to(script: str) -> set[str]:
    t = text().replace("\\\n", " ")
    flags = set()
    for m in re.finditer(rf"{re.escape(script)}(.*)", t):
        flags |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", m.group(1).split(">")[0]))
    return flags


def _flags_defined_in(path: Path) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', path.read_text()))


def test_flags_passed_to_the_trainer_exist():
    t = text().replace("\\\n", " ")
    body = t[t.index("trainer () {"):t.index("stage 3_check_both_arms")]
    passed = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", body))
    passed |= {f for a in ("ARM_A", "ARM_B") for f in _arm(a) if f.startswith("--")}
    passed |= {"--check-only", "--resume"}
    missing = passed - _flags_defined_in(REPO / "scripts" / "train_b1k_rooted.py")
    assert not missing, f"train_b1k_rooted.py has no {missing}"


def test_flags_passed_to_the_norm_stats_script_exist():
    """The first runner passed --config; the script takes --config-name. argparse
    would have rejected it at stage 1."""
    passed = _flags_passed_to("compute_norm_stats_b1k.py")
    assert passed, "norm stats call not found"
    missing = passed - _flags_defined_in(REPO / "scripts" / "compute_norm_stats_b1k.py")
    assert not missing, f"compute_norm_stats_b1k.py has no {missing}"


def test_flags_passed_to_the_data_scripts_exist():
    for script in ("slice_task_dataset.py", "merge_progress_labels.py"):
        passed = _flags_passed_to(script)
        missing = passed - _flags_defined_in(REPO / "scripts" / script)
        assert not missing, f"{script} has no {missing}"


# --- nothing important lives on the pod ----------------------------------------

def test_results_checkpoints_and_stats_live_on_the_volume():
    t = text()
    assert re.search(r'^VOL="\$\{VOL:-/workspace\}"', t, re.M)
    assert re.search(r'^RUN="\$\{RUN:-\$VOL/', t, re.M)
    assert re.search(r'^CKPT="\$RUN/', t, re.M) and re.search(r'^ASSETS="\$RUN/', t, re.M)
    for f in ("train_$name.log", "manifest.json", "gate.txt", "STATUS"):
        assert f'$RUN/{f}' in t or f'"$RUN/{f}"' in t, f


def test_preflight_refuses_container_disk_and_a_full_volume():
    t = text()
    assert "overlayfs" in t and "CKPT_NEED_GB" in t


def test_only_the_latest_checkpoint_is_kept():
    """openpi's default keep_period keeps six 10-16 GB checkpoints per arm at 30k."""
    assert "--keep-period 0" in text()


def test_a_resumed_run_refuses_data_that_changed():
    t = text()
    assert "data_fingerprint" in t and "resuming would change the training set" in t


# --- the pod stops itself --------------------------------------------------------

def test_there_is_an_hours_cap():
    t = text()
    assert re.search(r'MAX_HOURS="\$\{MAX_HOURS:-\d+\}"', t)
    assert "CAP_HIT" in t


def test_the_loader_gate_is_the_tested_one_and_it_can_stop_the_run():
    t = text()
    assert "analysis.stall_gate" in t and "GATE_NOGO" in t
    assert re.search(r'GATE_ENFORCE="\$\{GATE_ENFORCE:-1\}"', t), "enforced by default"


def test_a_supervisor_stop_reason_is_not_overwritten_by_the_failure_it_causes():
    t = text()
    fail = [l for l in t.splitlines() if l.startswith("fail  ()")][0]
    assert 'if [ -f "$RUN/STATUS" ]' in fail


# --- resuming, exit codes, credentials -----------------------------------------------

def test_training_resumes_and_a_finished_arm_is_skipped():
    t = text()
    assert "--resume" in t and "ALREADY_DONE" in t
    assert re.search(r'SAVE_EVERY="\$\{SAVE_EVERY:-(\d+)', t)
    assert int(re.search(r'SAVE_EVERY="\$\{SAVE_EVERY:-(\d+)', t).group(1)) <= 2000
    for line in t.splitlines():
        if "--resume" in line:
            assert "--overwrite" not in line


def test_the_trainers_own_exit_code_is_captured():
    t = text()
    assert "pipefail" in t
    assert re.search(r"^\s*local rc=\$\?", t, re.M)
    assert not any("PIPESTATUS" in l for l in code_lines())


def test_no_credential_touches_the_box():
    t = text()
    assert "login" not in t and "HF_TOKEN" not in t
    assert not re.search(r"hf_[A-Za-z0-9]{20}", t)


def test_it_records_a_manifest_of_what_was_run():
    t = text()
    for key in ('"steps"', '"lambda"', '"seed"', '"data_fingerprint"', '"norm_stats_sha"',
                '"config_a"', '"config_b"', '"gate"', '"openpi"'):
        assert key in t, key


# --- found preparing the first real launch, 2026-09-29 --------------------------------

def test_volume_free_space_is_measured_with_du_against_the_quota_not_df():
    """On a RunPod network volume `df` reports the whole shared VAST cluster (439 TB free
    was observed), not our quota (HANDOFF "How much room is actually left"). The df check
    would PASS on a volume with ~11 GB free, and training would die at its first 16 GB
    checkpoint -- possibly filling the volume that holds the evaluator env."""
    t = text()
    pre = t[t.index("stage 0_preflight"):t.index("stage 0_setup")]
    code = [l for l in pre.splitlines() if not l.lstrip().startswith("#")]
    assert not any("df " in l for l in code), "df cannot see a network volume's quota"
    assert "du " in pre and "VOL_QUOTA_GB" in pre
    assert re.search(r'VOL_QUOTA_GB="\$\{VOL_QUOTA_GB:-\d+\}"', t)


def test_checkpoint_space_need_covers_the_real_peak():
    """Peak during arm B: arm A's final + arm B's latest + one in flight, ~16 GB each."""
    m = re.search(r'CKPT_NEED_GB="\$\{CKPT_NEED_GB:-(\d+)\}"', text())
    assert m and int(m.group(1)) >= 48


def test_the_lr_schedule_decays_over_the_run_that_is_actually_trained():
    """Both arm configs use CosineDecaySchedule with a FIXED decay_steps=30_000. A 10k run
    would end with the LR still ~80% of peak, never annealed -- both arms weaker, the
    effect harder to see. The decay must follow STEPS."""
    t = text()
    body = t[t.index("trainer () {"):t.index("stage 3_check_both_arms")]
    assert "--lr-decay-steps $STEPS" in body, "in the SHARED call, so both arms get the same schedule"


def _supervisor() -> str:
    t = text()
    return t[t.index("supervise () {"):t.index("supervise & SUPERVISOR")]


def _run_arm() -> str:
    t = text()
    return t[t.index("run_arm () {"):t.index("stage 4_arm_A")]


def test_the_supervisor_runs_the_tested_health_gate_on_both_arms_and_it_can_stop_the_run():
    sup = _supervisor()
    assert "analysis.health_gate" in sup
    assert "for arm in A B" in sup and "--arm $arm" in sup
    assert "HEALTH_FAIL" in sup and "pkill -f train_b1k_rooted.py" in sup


def test_the_resume_path_is_exercised_once_before_it_is_depended_on():
    """A 23-65 h run depends on --resume, which had never run. The supervisor kills
    the trainer once, right after arm A's first checkpoint is FINALISED (orbax writes to a
    tmp dir and renames it; killing mid-save could corrupt it). The restart's first logged
    step must be >= that checkpoint's step, or the run stops: resume is broken."""
    t, sup = text(), _supervisor()
    assert re.search(r'RESUME_TEST="\$\{RESUME_TEST:-1\}"', t), "on by default"
    assert "orbax-checkpoint-tmp" in sup, "wait for the checkpoint to be finalised"
    assert "resume_test.killed" in sup
    assert "RESUME_TEST_RESTART" in sup and "RESUME_BROKEN" in sup


def test_a_crash_resumes_automatically_but_a_supervisor_stop_does_not():
    arm = _run_arm()
    assert re.search(r'MAX_RETRIES="\$\{MAX_RETRIES:-\d+\}"', text())
    assert "while" in arm, "retry loop"
    assert '[ -f "$RUN/STATUS" ]' in arm, "a deliberate stop (cap, gate, health) must not be retried"
    assert "AUTO_RESUME" in arm


def test_resume_is_decided_per_attempt_not_once():
    """The first attempt starts fresh; every retry must see the checkpoint dir."""
    arm = _run_arm()
    loop = arm[arm.index("while"):]
    assert "--resume" in loop


def test_the_restart_marker_is_written_before_the_resumed_trainer_logs_anything():
    """The supervisor verifies the resume by the first step logged AFTER the marker, so
    the marker must be written and then `continue` straight into the next attempt."""
    arm = _run_arm()
    i = arm.index("RESUME_TEST_RESTART")
    after = arm[i:].splitlines()[1].strip()
    assert after == "continue", after


def test_worker_count_follows_the_box():
    """24 workers is the prediction; on a smaller box, oversubscribing the vCPUs would
    make the loader slower, not faster. Auto = min(24, nproc - 3)."""
    t = text()
    assert re.search(r'WORKERS="\$\{WORKERS:-auto\}"', t)
    assert "nproc" in t and "- 3" in t


def test_the_manifest_records_the_box_and_every_verdict():
    t = text()
    man = t[t.index("stage 6_manifest"):]
    for key in ('"vcpus"', '"gpu"', '"resume_test"', '"health_armA"', '"health_armB"', '"gate"'):
        assert key in man, key


# --- the pod CANNOT stop itself (verified 2026-09-30) -----------------------------------
# RunPod's injected per-pod RUNPOD_API_KEY returns 403 for get, stop and terminate on its
# own pod (runpodctl v1.14.15 and v2.14.0 both; RunPod docs and community reports agree).
# The only in-pod fix is an account-level key on the rented box, which the project
# forbids. So the pod writes a TERMINAL STATUS on every exit path and a watchdog on the
# owner's laptop (scripts/watchdog.sh) terminates it. The earlier tests here asserted the
# pod could stop itself; they were wrong and are replaced.

def test_every_exit_path_leaves_a_terminal_status_for_the_watchdog():
    t = text()
    assert "trap finish EXIT" in t
    fin = t[t.index("finish () {"):t.index("trap finish EXIT")]
    assert 'status "EXITED rc=$rc"' in fin, "an exit with no recorded reason still leaves STATUS"
    assert "TERMINAL" in fin, "the watchdog keys on one marker, not a list of reasons"


def test_the_runner_does_not_pretend_the_pod_can_stop_itself():
    code = "\n".join(code_lines())
    assert "runpodctl stop pod" not in code and "runpodctl remove pod" not in code
    assert "runpodctl get pod" not in code, "preflight must not require a permission that does not exist"


def test_the_runner_names_its_pod_for_the_watchdog():
    """The watchdog terminates exactly the pod the run is on -- recorded, not guessed."""
    t = text()
    assert "RUNPOD_POD_ID" in t and "$RUN/pod_id" in t
