"""scripts/watchdog.sh -- the only thing that can stop the training pod.

RunPod's per-pod key cannot stop its own pod (403, verified 2026-09-30), so the pod
writes TERMINAL to $RUN/STATUS and this watchdog, on the owner's laptop, terminates it.
If the watchdog is wrong, the pod bills until the balance is gone and the volume holding
the checkpoints is at risk. So it is EXECUTED here against a fake ssh and a fake
runpodctl, not read.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WD = REPO / "scripts" / "watchdog.sh"
SECRET = "rpa_TESTKEY_should_never_leak_0123456789"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(tmp: Path, status: str, pod_exists: bool = True, deadline_in: int = 3600,
         key_mode: int = 0o600, loops: int = 1, age: int = 60) -> dict:
    bin_ = tmp / "bin"
    bin_.mkdir()
    # fake ssh: prints what a PTY session would, including the STATUS content
    (bin_ / "ssh").write_text(f'''#!/usr/bin/env bash
cat > /dev/null
echo "__RPSTART__"
printf '%s\\n' "{status}"
echo "LOGAGE {age}"
echo "__RPEND__"
''')
    # fake runpodctl: records argv and whether the key arrived via env (never via argv)
    (bin_ / "runpodctl").write_text(f'''#!/usr/bin/env bash
echo "argv: $*" >> {tmp}/runpodctl.calls
[ "${{RUNPOD_API_KEY:-}}" = "{SECRET}" ] && echo "env-key: yes" >> {tmp}/runpodctl.calls
case "$1 $2" in
  "pod get")    {"echo '{\"id\":\"POD\"}'" if pod_exists else "echo 'api request failed with status 404' >&2; exit 1"} ;;
  "pod delete") echo deleted ;;
esac
''')
    for f in ("ssh", "runpodctl"):
        (bin_ / f).chmod(0o755)
    key = tmp / "watchdog.key"
    key.write_text(SECRET)
    key.chmod(key_mode)
    return {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "POD_ID": "POD",
            "SSH_USER": "POD-123", "KEY_FILE": str(key), "INTERVAL": "0",
            "DEADLINE_EPOCH": str(int(time.time()) + deadline_in), "MAX_LOOPS": str(loops),
            "LOG": str(tmp / "wd.log"), "HOME": str(tmp)}


def _run(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(WD), *args], env=env, capture_output=True, text=True, timeout=60)


def _calls(tmp: Path) -> str:
    f = tmp / "runpodctl.calls"
    return f.read_text() if f.exists() else ""


def test_terminal_status_terminates_the_pod(tmp_path):
    p = _run(_env(tmp_path, "DONE 2026-10-02T00:00Z\nTERMINAL 2026-10-02T00:00Z"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "argv: pod delete POD" in _calls(tmp_path)


def test_a_running_job_is_left_alone(tmp_path):
    p = _run(_env(tmp_path, "", loops=3))
    assert "pod delete" not in _calls(tmp_path), p.stdout + p.stderr


def test_a_failure_status_without_terminal_is_not_yet_acted_on(tmp_path):
    """The supervisor writes the reason, then the EXIT trap appends TERMINAL. Acting on
    the reason alone could kill the pod before its logs are synced."""
    _run(_env(tmp_path, "GATE_NOGO mean 7.9 s/step"))
    assert "pod delete" not in _calls(tmp_path)


def test_the_deadline_is_a_backstop_even_if_the_run_never_ends(tmp_path):
    _run(_env(tmp_path, "", deadline_in=-10))
    assert "argv: pod delete POD" in _calls(tmp_path)


def test_a_pod_that_is_already_gone_ends_the_watch_quietly(tmp_path):
    p = _run(_env(tmp_path, "", pod_exists=False, loops=5))
    assert p.returncode == 0
    assert "pod delete" not in _calls(tmp_path)
    assert "gone" in (tmp_path / "wd.log").read_text().lower()


def test_the_key_travels_only_in_the_environment_and_never_into_logs(tmp_path):
    _run(_env(tmp_path, "TERMINAL x"))
    calls = _calls(tmp_path)
    assert "env-key: yes" in calls
    assert SECRET not in calls.replace("env-key: yes", ""), "the key must not be in argv"
    assert SECRET not in (tmp_path / "wd.log").read_text()


def test_a_key_file_others_can_read_is_refused(tmp_path):
    p = _run(_env(tmp_path, "TERMINAL x", key_mode=0o644))
    assert p.returncode != 0 and "600" in (p.stdout + p.stderr)
    assert "pod delete" not in _calls(tmp_path)


def test_check_mode_proves_access_but_never_terminates(tmp_path):
    p = _run(_env(tmp_path, "TERMINAL x"), "check")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "argv: pod get POD" in _calls(tmp_path)
    assert "pod delete" not in _calls(tmp_path)


# --- the progress backstop (2026-10-01: a deadlocked run wrote nothing for 14.5 h) -------

def test_a_run_silent_for_longer_than_the_stall_limit_is_terminated(tmp_path):
    """The pod's own supervisor kills a hung trainer after 30 min and resumes. If the
    supervisor itself is dead, nothing on the pod notices; at 90 min of silence the
    watchdog stops the billing."""
    _run(_env(tmp_path, "", age=100 * 60))
    calls = _calls(tmp_path)
    assert "argv: pod delete POD" in calls
    assert "silent" in (tmp_path / "wd.log").read_text().lower()


def test_a_run_that_is_writing_is_left_alone(tmp_path):
    _run(_env(tmp_path, "", age=120, loops=2))
    assert "pod delete" not in _calls(tmp_path)


def test_the_age_line_is_not_mistaken_for_status(tmp_path):
    """LOGAGE shares the SSH output with STATUS; it must never read as a run verdict."""
    _run(_env(tmp_path, "", age=60))
    log = (tmp_path / "wd.log").read_text()
    assert "LOGAGE" not in [l.split("ok: ")[-1].split()[0] for l in log.splitlines() if "ok: " in l]
