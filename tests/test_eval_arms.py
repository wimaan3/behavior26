"""scripts/eval_arms.sh -- evaluate arm A vs arm B, paired, on the frozen dev subset.

Evaluation is where a quiet mistake turns into a wrong RESULT rather than a crash: an
unnormalised policy, a different instance list per arm, or a test-set instance would all
produce a number that reads as the treatment's effect. These tests pin what makes the
number mean what it claims, and EXECUTE the pieces that can run on a laptop.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "scripts" / "eval_arms.sh"
CFG = REPO / "configs" / "experiments" / "001-dev-loop.yaml"


def text() -> str:
    return EVAL.read_text()


def code() -> str:
    return "\n".join(l for l in text().splitlines() if not l.lstrip().startswith("#"))


def test_it_exists_and_is_bash():
    assert text().startswith("#!/usr/bin/env bash")


# --- the comparison must be paired and on the frozen subset ---------------------------

def test_tasks_instances_mode_and_rollouts_come_from_the_frozen_config():
    """One source of truth. A copy of the instance list in the script could drift from
    configs/experiments/001-dev-loop.yaml and silently change what is being compared."""
    t = text()
    assert "configs/experiments/001-dev-loop.yaml" in t
    for key in ("instances", "tasks", "mode", "num_rollouts"):
        assert key in t, key
    assert "seq 10 36" not in code(), "do not hard-code the frozen list"


def test_both_arms_run_one_shared_evaluator_call():
    """Both arms through the same call means the same instances, mode, wrapper and
    rollout count by construction."""
    assert code().count("omnigibson.eval.eval") == 1


def test_training_instances_only_and_a_test_instance_is_refused():
    t = text()
    assert "--mode train" in t or '"$MODE"' in t
    assert "301" in t and "refus" in t.lower()


def test_units_run_task_major_with_both_arms_back_to_back():
    """If money or time runs out midway, what is finished must be complete PAIRS: arm A
    and arm B on the same task, not all of arm A and none of arm B."""
    t = text()
    loop = t[t.index("for TASK in"):]
    assert loop.index("for ARM in") < loop.index("done")


def test_a_unit_counts_as_done_only_with_every_rollout():
    t = text()
    assert "EXPECTED" in t and ".done" in t


# --- what is served must be what was trained -------------------------------------------

def test_both_checkpoints_are_validated_for_the_exact_stats_file_before_anything_runs():
    t = text()
    pre = t[t.index("stage 0_preflight"):t.index("stage 1_setup")]
    assert "assets/$ASSET_ID/norm_stats.json" in pre


def test_the_server_is_told_where_the_stats_are():
    assert "ASSET_ID=" in text() and "serve_baseline.sh" in text()


def test_the_full_patch_set_is_applied_because_arm_b_needs_patch_0002():
    """--compat-only (the baseline's mode) lacks the progress-head config; serving arm B
    would fail after Isaac Sim had paid its scene load."""
    t = code()
    assert "apply_openpi_patches.sh" in t and "--compat-only" not in t


def test_each_checkpoint_answers_one_real_inference_before_the_simulator_starts():
    t = text()
    smoke = t.index("stage 2_smoke")
    assert smoke < t.index("stage 3_eval")
    body = t[smoke:t.index("stage 3_eval")]
    assert "create_trained_policy" in body and "SMOKE_OK" in body
    # NOT openpi's make_b1k_example: it carries a 23-number state, but B1KInputs indexes
    # the full proprio vector at the robot config's positions (53+). The first eval pod
    # failed its smoke test on exactly that -- a wrong input, not a broken checkpoint.
    assert "make_b1k_example()" not in body
    assert "robot_config" in body and ".proprio" in body and ".observations" in body


def test_the_server_is_killed_by_process_group_between_units():
    """The 2026-10-01 deadlock: killing a parent left children holding resources. A
    surviving server would also keep port 8000, and the next unit would talk to the
    PREVIOUS arm's policy."""
    t = code()
    assert "setsid" in t and "kill -TERM --" in t


def test_the_next_unit_refuses_to_start_while_the_old_server_still_answers():
    t = text()
    assert "port 8000 still answering" in t.lower() or "still serving" in t.lower()


# --- results, video, and the pod ------------------------------------------------------

def test_results_go_to_the_volume_and_videos_are_compacted_under_a_space_guard():
    t = text()
    assert re.search(r'OUT="\$\{OUT:-/workspace/', t)
    assert "--write-video" in t
    assert "VIDEO_KEEP_FREE_GB" in t and "ffmpeg" in t


def test_the_paired_comparison_runs_at_the_end():
    assert "analysis/compare.py" in text() or "analysis.compare" in text()


def test_every_exit_leaves_a_terminal_status_for_the_watchdog():
    t = text()
    assert "trap finish EXIT" in t and "TERMINAL" in t and "$OUT/pod_id" in t


def test_there_is_an_hours_cap():
    assert re.search(r'MAX_HOURS="\$\{MAX_HOURS:-\d+\}"', text())


# --- executed: the config reader --------------------------------------------------------

@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_config_reader_returns_the_frozen_values():
    t = text()
    m = re.search(r"read_config \(\) \{.*?\n\}", t, re.S)
    assert m, "read_config function"
    script = f"set -uo pipefail\nPY=python3\nB26=\"{REPO}\"\nEXPERIMENT=configs/experiments/001-dev-loop.yaml\n{m.group(0)}\nread_config\n" \
             'echo "TASKS=${TASKS[*]}|MODE=$MODE|N=${#INSTANCES[@]}|FIRST=${INSTANCES[0]}|LAST=${INSTANCES[-1]}|ROLLOUTS=$ROLLOUTS"'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "TASKS=set_up_a_coffee_station_in_your_kitchen putting_shoes_on_rack" in p.stdout
    assert "MODE=train|N=27|FIRST=10|LAST=36|ROLLOUTS=1" in p.stdout, p.stdout


def test_the_installer_can_skip_the_released_baseline_download():
    """Evaluating OUR arms does not need the released checkpoint; its Google-Drive
    download is slow and a failure there would abort the whole install."""
    inst = (REPO / "scripts" / "install_openpi.sh").read_text()
    assert 'SKIP_BASELINE' in inst
    i = inst.index("STAGE 4")
    assert "SKIP_BASELINE" in inst[i - 200:i + 600]


def test_the_served_prompt_is_checked_against_the_training_prompt_before_the_simulator():
    """The first eval pod's servers died on KeyError: TASK_REGISTRY lacked our tasks. And a
    registered-but-wrong prompt would not crash at all -- it would quietly handicap both
    arms. Check, before Isaac Sim, that each task's registry prompt is the task name the
    arms were trained on (prompt_from_task + tasks_from_metadata)."""
    t = text()
    smoke = t[t.index("stage 2_smoke"):t.index("stage 3_eval")]
    assert "TASK_REGISTRY" in smoke and "PROMPTS_OK" in smoke


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_evaluator_receives_every_argument(tmp_path):
    """The second eval pod died on "the following arguments are required: --task-name":
    the launch used bash -c '... && shift && exec python ... "$@"' "$VOL" --task-name ...,
    and in bash -c the first extra argument is $0, so `shift` discarded --task-name. This
    runs the REAL launch block with python replaced by a stub that prints what it got."""
    t = text()
    start = t.rindex("\n", 0, t.index("setsid bash -c")) + 1     # the whole line, prefix included
    end = t.index("--write-video --headless", start) + len("--write-video --headless")
    block = t[start:end]
    (tmp_path / "env.sh").write_text('python () { echo "ARGS: $*"; }\n')
    script = f'''
VOL="{tmp_path}"; TASK=t1; MODE=train; ROLLOUTS=1; WRAPPER=w; SCR="{tmp_path}"; ARM=A
INSTANCES=(10 11)
{block} 2>&1
'''
    script = script.replace("exec python", "python")
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    out = p.stdout + p.stderr
    assert "ARGS: -m omnigibson.eval.eval --task-name t1" in out, out
    assert "--instance-indices 10 11" in out and "--write-video --headless" in out
