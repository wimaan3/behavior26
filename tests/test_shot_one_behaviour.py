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
kill_trainer () {{ :; }}
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


# --- the 14.5-hour deadlock of 2026-10-01 ---------------------------------------------
# The resume test SIGTERM'd the trainer python. Its multiprocessing helpers (dataloader
# workers, resource_tracker) survived, were re-parented to init, and kept the trainer's
# stdout pipe open. The timestamping `while read` never saw EOF, run_arm never reached its
# retry, and the pod idled for 14.5 h. The fake trainer above has no children, so it could
# not catch this. These tests use a trainer that DOES spawn a child holding the pipe.

def _fn(name: str) -> str:
    start = SHOT.index(f"{name} () {{")
    depth, i = 0, SHOT.index("{", start)
    while True:
        if SHOT[i] == "{":
            depth += 1
        elif SHOT[i] == "}":
            depth -= 1
            if depth == 0:
                return SHOT[start:i + 1]
        i += 1


def _pipeline_script(tmp: Path, then: str) -> str:
    """The REAL trainer() with $PY replaced by a stub that spawns a long-lived child (as
    the dataloader does) and then waits -- run through the same timestamp pipe as run_arm."""
    stub = tmp / "fakepy"
    stub.write_text("#!/usr/bin/env bash\nsleep 300 &\necho 'Step 1: action_loss=0.5'\nsleep 300\n")
    stub.chmod(0o755)
    (tmp / "b26" / "scripts").mkdir(parents=True)
    return f'''
set -uo pipefail
RUN={tmp}; B26={tmp}/b26; PY={stub}; ROOT=r; ASSETS=a; CKPT=c; STEPS=10; BATCH=1; WORKERS=1
SEED=0; SAVE_EVERY=5; LOG_EVERY=1; TASKS=(t); TSTAMP='while IFS= read -r l; do echo "$l"; done'
{_fn("trainer")}
{_fn("kill_trainer")}
{_fn("reap_orphans") if "reap_orphans () {" in SHOT else ""}
( trainer armA cfgA 2>&1 | eval "$TSTAMP" ) > {tmp}/out.log &
PIPE=$!
for i in $(seq 1 50); do [ -s {tmp}/trainer.pgid ] && grep -q "Step 1" {tmp}/out.log && break; sleep 0.1; done
{then}
for i in $(seq 1 100); do kill -0 $PIPE 2>/dev/null || {{ echo PIPELINE_ENDED; exit 0; }}; sleep 0.1; done
echo PIPELINE_HUNG; kill -KILL -- -$(cat {tmp}/trainer.pgid) 2>/dev/null; exit 1
'''


def test_the_trainer_runs_as_its_own_process_group(tmp_path):
    t = _fn("trainer")
    assert 'echo $BASHPID > "$RUN/trainer.pgid"' in t and "exec setsid" in t


def test_killing_the_trainer_takes_its_children_and_frees_the_pipe(tmp_path):
    p = subprocess.run(["bash", "-c", _pipeline_script(tmp_path, "kill_trainer")],
                       capture_output=True, text=True, timeout=60)
    assert "PIPELINE_ENDED" in p.stdout, p.stdout + p.stderr


def test_killing_only_the_leader_reproduces_the_deadlock(tmp_path):
    """The control: what the old supervisor did (kill the python alone) leaves the child
    holding the pipe, and the pipeline hangs. If this ever passes, the test above is not
    testing anything."""
    p = subprocess.run(["bash", "-c", _pipeline_script(tmp_path, 'kill -TERM $(cat ' + str(tmp_path) + '/trainer.pgid)')],
                       capture_output=True, text=True, timeout=60)
    assert "PIPELINE_HUNG" in p.stdout, p.stdout + p.stderr


def test_orphans_of_a_trainer_that_died_on_its_own_are_reaped(tmp_path):
    """A crash, not a kill: the leader exits, its child keeps the pipe. reap_orphans must
    see a dead leader with live group members and kill the group."""
    then = 'kill -KILL $(cat ' + str(tmp_path) + '/trainer.pgid); sleep 0.3; ORPHAN_GRACE_LOOPS=1; reap_orphans; reap_orphans'
    p = subprocess.run(["bash", "-c", _pipeline_script(tmp_path, then)],
                       capture_output=True, text=True, timeout=60)
    assert "PIPELINE_ENDED" in p.stdout, p.stdout + p.stderr


def test_the_supervisor_kills_by_group_everywhere_and_never_by_name():
    sup = SHOT[SHOT.index("supervise () {"):SHOT.index("supervise & SUPERVISOR")]
    code = "\n".join(l for l in sup.splitlines() if not l.lstrip().startswith("#"))
    assert "pkill -f train_b1k_rooted.py" not in code
    assert code.count("kill_trainer") >= 4, "cap, gate, health, resume test, stall"
    assert "reap_orphans" in code


def test_a_run_with_no_new_step_for_too_long_is_treated_as_hung():
    t = SHOT
    assert re.search(r'STALL_MINUTES="\$\{STALL_MINUTES:-\d+\}"', t)
    sup = t[t.index("supervise () {"):t.index("supervise & SUPERVISOR")]
    assert "STALL_MINUTES" in sup and "HUNG" in sup


def test_a_fresh_launch_that_resumes_is_verified_too(tmp_path):
    """Shot one's relaunch after the deadlock resumes arm A on a NEW pod from the step-1000
    checkpoint on the volume -- a stronger resume test than the in-run kill. The pending
    verification must see it: run_arm logs the restart marker whenever it resumes while
    the resume test is unverified, and passes --resume."""
    run = tmp_path / "run"
    (run / "ck" / "cfgA" / "armA" / "1000").mkdir(parents=True)
    rc, out, run = _run_existing(run, "ok", pre=f"date > {run}/resume_test.killed")
    log = (run / "train_armA.log").read_text()
    assert "RESUME_TEST_RESTART" in log
    first = [l for l in log.splitlines() if l.startswith("call 1 args:")][0]
    assert "--resume" in first
    assert log.index("RESUME_TEST_RESTART") < log.index("call 1 args:")


def test_no_marker_once_the_resume_test_has_a_verdict(tmp_path):
    run = tmp_path / "run"
    (run / "ck" / "cfgA" / "armA" / "1000").mkdir(parents=True)
    pre = f"date > {run}/resume_test.killed; echo 'RESUME_TEST PASS' > {run}/resume_test.txt"
    rc, out, run = _run_existing(run, "ok", pre=pre)
    assert "RESUME_TEST_RESTART" not in (run / "train_armA.log").read_text()


def _run_existing(run: Path, scenario: str, pre: str = ""):
    """Like _run, but into a RUN dir the test prepared (a checkpoint already there)."""
    script = f'''
set -uo pipefail
RUN={run}; CKPT={run}/ck; STEPS=10; WORKERS=1; MAX_RETRIES=2; SCENARIO={scenario}
TSTAMP='cat'
nvidia-smi () {{ echo 0,0; }}
sleep () {{ command sleep 0.05; }}
kill_trainer () {{ :; }}
{FAKE}
{pre}
{_run_arm_source()}
run_arm armA cfgA
echo "RC=$?"
'''
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr, run
