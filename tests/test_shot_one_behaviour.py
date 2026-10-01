"""shot_one.sh's control flow, EXECUTED -- not read.

test_shot_one.py pins the runner's text. That cannot catch a retry loop that retries a
deliberate stop, or a resume check whose awk reads the wrong step. These tests lift the
real run_arm() and the real resume-verification awk out of the script and run them in
bash against a fake trainer, so a mistake shows up here instead of on a paid pod.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHOT = (REPO / "scripts" / "shot_one.sh").read_text()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run_arm_source() -> str:
    start = SHOT.index("run_arm () {")
    end = SHOT.index("stage 4_arm_A")
    return SHOT[start:end]


# The fake trainer: behaviour chosen per call from $SCENARIO, call count kept in a file.
FAKE = r'''
trainer () {
  local n; n=$(( $(cat "$RUN/calls" 2>/dev/null || echo 0) + 1 )); echo $n > "$RUN/calls"
  echo "call $n args: $*"
  mkdir -p "$CKPT/$2/$1"
  case "$SCENARIO:$n" in
    crash_then_ok:1) return 1 ;;
    crash_then_ok:*) return 0 ;;
    status_stop:*)   echo "GATE_NOGO x" > "$RUN/STATUS"; return 1 ;;
    resume_test:1)   return 1 ;;
    resume_test:*)   echo "Step 1000: action_loss=0.3"; return 0 ;;
    always_crash:*)  return 1 ;;
    ok:*)            return 0 ;;
  esac
}
'''


def _run(tmp_path: Path, scenario: str, pre: str = "") -> tuple[int, str, Path]:
    run = tmp_path / "run"
    run.mkdir()
    script = f'''
set -uo pipefail
RUN={run}; CKPT={run}/ck; STEPS=10; WORKERS=1; MAX_RETRIES=2; SCENARIO={scenario}
TSTAMP='cat'
nvidia-smi () {{ echo 0,0; }}
sleep () {{ command sleep 0.05; }}
{FAKE}
{pre}
{_run_arm_source()}
run_arm armA cfgA
echo "RC=$?"
'''
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr, run


def _calls(run: Path) -> int:
    return int((run / "calls").read_text())


def test_a_crash_is_resumed_and_the_retry_passes_resume(tmp_path):
    rc, out, run = _run(tmp_path, "crash_then_ok")
    assert "RC=0" in out and (run / "armA.done").exists()
    assert _calls(run) == 2
    log = (run / "train_armA.log").read_text()
    assert "AUTO_RESUME attempt 1/2" in log
    second = [l for l in log.splitlines() if l.startswith("call 2 args:")][0]
    assert "--resume" in second, "the retry must resume from the checkpoint"
    first = [l for l in log.splitlines() if l.startswith("call 1 args:")][0]
    assert "--resume" not in first, "the first attempt starts fresh"


def test_a_deliberate_stop_is_never_retried(tmp_path):
    rc, out, run = _run(tmp_path, "status_stop")
    assert "RC=1" in out and not (run / "armA.done").exists()
    assert _calls(run) == 1, "a gate/health/cap stop must not be undone by a retry"


def test_the_resume_test_kill_restarts_without_spending_a_retry(tmp_path):
    rc, out, run = _run(tmp_path, "resume_test", pre=f'touch {tmp_path}/run/resume_test.killed')
    assert "RC=0" in out and (run / "armA.done").exists()
    log = (run / "train_armA.log").read_text()
    assert "RESUME_TEST_RESTART" in log and "AUTO_RESUME" not in log
    assert log.index("RESUME_TEST_RESTART") < log.index("Step 1000:"), \
        "the marker must precede the resumed trainer's output"


def test_a_permanent_crash_gives_up_after_max_retries(tmp_path):
    rc, out, run = _run(tmp_path, "always_crash")
    assert "RC=1" in out and not (run / "armA.done").exists()
    assert _calls(run) == 3, "1 attempt + MAX_RETRIES=2"


def test_a_finished_arm_is_skipped(tmp_path):
    rc, out, run = _run(tmp_path, "ok", pre=f'date > {tmp_path}/run/armA.done')
    assert "ALREADY_DONE armA" in out and not (run / "calls").exists()


# --- the resume verification the supervisor runs --------------------------------------

def _resume_awk() -> str:
    sup = SHOT[SHOT.index("supervise () {"):SHOT.index("supervise & SUPERVISOR")]
    m = re.search(r"awk '(.*?)'\)", sup, re.S)
    assert m, "resume-check awk not found"
    return m.group(1)


def _first_step_after_marker(tmp_path: Path, log: str) -> str:
    f = tmp_path / "train.log"
    f.write_text(log)
    p = subprocess.run(["bash", "-c", f"tr '\\r' '\\n' < {f} | awk '{_resume_awk()}'"],
                       capture_output=True, text=True, timeout=30)
    return p.stdout.strip()


def test_the_resume_check_reads_the_first_step_after_the_restart(tmp_path):
    log = ("1.0 \r\rStep 998: action_loss=0.2\n1.1 \r\rStep 999: action_loss=0.2\n"
           "1.2 === RESUME_TEST_RESTART (deliberate kill)\n"
           "1.3 \r\rStep 1000: action_loss=0.2\n1.4 \r\rStep 1001: action_loss=0.2\n")
    assert _first_step_after_marker(tmp_path, log) == "1000"


def test_the_resume_check_catches_a_restart_from_zero(tmp_path):
    log = ("1.0 \r\rStep 1003: action_loss=0.2\n1.2 === RESUME_TEST_RESTART\n"
           "1.3 \r\rStep 0: action_loss=0.9\n")
    assert _first_step_after_marker(tmp_path, log) == "0"


def test_the_resume_check_waits_while_nothing_is_logged_after_the_restart(tmp_path):
    log = "1.0 \r\rStep 1003: action_loss=0.2\n1.2 === RESUME_TEST_RESTART\n1.3 loading model\n"
    assert _first_step_after_marker(tmp_path, log) == ""
