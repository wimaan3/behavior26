"""The readiness probe, pinned against the false positive it was written for.

A probe that says SERVER_DIED about a healthy server is worse than no probe: it
kills a run that was working, and on rented hardware it costs a scene load to
find out. Measured 2026-09-14: the naive loop fired at t=10s while the server
went on to restore a 6.2 GiB checkpoint and serve normally, because for the
first seconds nothing on the box had "serve_b1k" in its argv yet.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WAIT = REPO / "scripts" / "wait_for_policy_server.sh"


def _run(env_extra: dict, timeout: int = 30) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, **{k: str(v) for k, v in env_extra.items()})
    return subprocess.run(["bash", str(WAIT)], capture_output=True, text=True,
                          env=env, timeout=timeout)


def test_a_missing_process_inside_the_grace_period_is_not_death():
    """The exact false positive. Nothing matches the pattern and nothing serves,
    but within GRACE the probe must keep waiting rather than declare death."""
    t0 = time.monotonic()
    res = _run({"PORT": "59999", "PATTERN": "definitely-not-running-xyz",
                "GRACE": 6, "TIMEOUT": 10})
    elapsed = time.monotonic() - t0
    # 1s tolerance: the script measures with integer `date +%s`, which can report
    # ELAPSED=6 when only 5.05s of wall time have passed. Without the tolerance
    # this test fails intermittently under load, which is worse than useless --
    # a flaky guard gets deleted.
    assert "SERVER_DIED" not in res.stdout or elapsed >= 5, (
        f"declared death after {elapsed:.1f}s, inside the 6s grace: {res.stdout}"
    )


def test_death_is_still_detected_after_the_grace_period():
    """The grace must not become a blanket excuse -- a genuinely absent server
    still has to be reported, and before TIMEOUT."""
    res = _run({"PORT": "59999", "PATTERN": "definitely-not-running-xyz",
                "GRACE": 2, "TIMEOUT": 25})
    assert "SERVER_DIED" in res.stdout, res.stdout + res.stderr


def test_the_launcher_pattern_is_matched_not_just_the_final_process():
    """`bash serve_baseline.sh` execs uv which spawns python; during the handover
    only the launcher name exists. Probing one name alone reintroduces the bug."""
    text = WAIT.read_text()
    assert "serve_baseline" in text and "serve_b1k" in text


def test_an_ancestor_matching_the_pattern_is_not_mistaken_for_the_server():
    """The silent half of the bug. `pgrep -f` matches any PARENT whose command
    line contains the pattern -- the launching shell, a CI wrapper, an inline
    `PATTERN=... bash wait...`. That probe reports alive forever and can never
    report death. Reproduced in this very test harness, which wraps commands in
    a shell whose argv contains the pattern verbatim.
    """
    text = WAIT.read_text()
    assert "ancestors" in text and "server_alive" in text, (
        "liveness must exclude self and ancestors"
    )
    # Launch with the pattern present in OUR OWN command line; death must still
    # be detected, because the only match is an ancestor.
    # shlex.quote: the repo path contains a space on the development machine, and
    # an unquoted interpolation splits it into two arguments. Bitten before.
    import shlex
    res = subprocess.run(
        ["bash", "-c",
         "PATTERN=zzz-ancestor-marker PORT=59999 GRACE=2 TIMEOUT=20 "
         f"bash {shlex.quote(str(WAIT))}"],
        capture_output=True, text=True, timeout=40)
    assert "SERVER_DIED" in res.stdout, (
        f"an ancestor match masked a dead server: {res.stdout}{res.stderr}"
    )
