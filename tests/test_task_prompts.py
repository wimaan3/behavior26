"""The prompt served at evaluation must be the prompt the arms were trained with.

Both arms trained with prompt_from_task=True, and openpi's tasks_from_metadata maps each
task_index to the task NAME in the dataset's meta/tasks.parquet (index 10 ->
"set_up_a_coffee_station_in_your_kitchen", 22 -> "putting_shoes_on_rack"; read from the
dataset on 2026-10-03). serve_b1k.py, however, takes the prompt from TASK_REGISTRY, which
upstream holds only turning_on_radio -- the first eval pod died on KeyError for both of
our tasks. Patch 0003 registers them, mapping each name to ITSELF: a natural-language
instruction would be a prompt the models never saw.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PATCH = next((REPO / "training" / "patches").glob("0003-*.patch"), None)
APPLY = REPO / "scripts" / "apply_openpi_patches.sh"


def test_patch_0003_registers_every_dev_task_with_its_training_prompt():
    assert PATCH is not None, "training/patches/0003-*.patch"
    added = [l[1:] for l in PATCH.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]
    pairs = dict(re.findall(r'"([a-z_]+)":\s*"([^"]+)"', "\n".join(added)))
    tasks = yaml.safe_load((REPO / "configs/experiments/001-dev-loop.yaml").read_text())["tasks"]
    for t in tasks:
        assert pairs.get(t) == t, f"{t} must be served the prompt it was trained on: {t!r}"


def test_patch_0003_touches_only_the_task_registry():
    files = re.findall(r"^\+\+\+ b/(\S+)", PATCH.read_text(), re.M)
    assert files == ["src/openpi/configs/tasks/b1k.py"]


def test_the_full_patch_set_includes_0003():
    t = APPLY.read_text()
    assert "000[2-9]" in t or "0003" in t


# --- executed: patches apply one at a time, so a pod with 0001+0002 can take 0003 --------

@pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="git+bash")
def test_a_partially_patched_checkout_takes_only_the_missing_patch(tmp_path):
    repo = tmp_path / "openpi"; repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    for f in ("a.txt", "b.txt", "c.txt"):
        (repo / f).write_text("base\n")
    git("add", "."); git("commit", "-qm", "base")
    pdir = tmp_path / "patches"; pdir.mkdir()
    for n, f in (("0001-a", "a.txt"), ("0002-b", "b.txt"), ("0003-c", "c.txt")):
        (repo / f).write_text("base\npatched\n")
        (pdir / f"{n}.patch").write_text(git("diff"))
        git("checkout", "-q", "--", f)
    # the state of a pod that already ran the old script: 0001 and 0002 in, 0003 not
    git("apply", str(pdir / "0001-a.patch")); git("apply", str(pdir / "0002-b.patch"))
    env = {"PATH": "/usr/bin:/bin", "OPENPI_ROOT": str(repo), "PATCH_DIR": str(pdir), "HOME": str(tmp_path)}
    p = subprocess.run(["bash", str(APPLY)], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (repo / "c.txt").read_text() == "base\npatched\n", "0003 must be applied"
    assert (repo / "a.txt").read_text() == "base\npatched\n", "0001 must not be applied twice"
    p2 = subprocess.run(["bash", str(APPLY)], env=env, capture_output=True, text=True, timeout=60)
    assert p2.returncode == 0 and "already applied" in (p2.stdout + p2.stderr).lower()
