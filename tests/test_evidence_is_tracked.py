"""Results that live only on one laptop are not saved.

`*.log` was gitignored for rollout noise, and it silently caught every training log
under docs/ as well -- including train_rung4_armA.log, the only evidence for the
rung-4 prefetch-sawtooth finding. Commit messages said "logs in git" and they were
not. The docs/ tree is the project's record: nothing under it may be ignored.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(),
                                reason="needs a git checkout")


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO)
    return r.returncode == 0


@pytest.mark.parametrize("path", [
    "docs/sessionB-2026-09-16-rung2-5/logs/train_rung4_armA.log",
    "docs/any-future-session/logs/train_armA.log",
    "docs/any-future-session/eval.log",
    "docs/any-future-session/norm_stats.json",
])
def test_nothing_under_docs_is_ignored(path):
    assert not _ignored(path), f"{path} would never reach GitHub"


def test_scratch_logs_elsewhere_are_still_ignored():
    """The rule exists for rollout noise; keep it for everything outside docs/."""
    assert _ignored("rollouts/mockA/eval.log")
    assert _ignored("some_scratch_run.log")


def test_no_file_under_docs_is_currently_ignored():
    out = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "docs"],
                         cwd=REPO, capture_output=True, text=True).stdout.split()
    out = [p for p in out if "__pycache__" not in p]
    assert not out, f"on disk but ignored, so never pushed: {out}"
