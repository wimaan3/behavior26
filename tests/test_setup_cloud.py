"""
The cloud provisioning script, pinned where it is load-bearing.

Two expensive things must land on the network volume and not on container disk:
the conda env, and the 29.3 GB BEHAVIOR-1K dataset. If they land on container
disk the pod's death destroys both, and the A5000-vs-4090 comparison -- which
decides the card for ALL evaluation, our dominant cost line -- becomes a
rebuild instead of a remount.

These tests run without a GPU or a network by stubbing `nvidia-smi` and `git`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "scripts" / "setup_cloud.sh"


def _run(volume_root: Path | str, *, env_extra: dict | None = None,
         stub_bin: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["VOLUME_ROOT"] = str(volume_root)
    if stub_bin:
        env["PATH"] = f"{stub_bin}{os.pathsep}{env['PATH']}"
    env.update(env_extra or {})
    return subprocess.run(["bash", str(SETUP)], capture_output=True, text=True, env=env)


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    """Stand-ins for the two commands that need real hardware or the network."""
    d = tmp_path / "stub-bin"
    d.mkdir()
    for name in ("nvidia-smi", "git"):
        p = d / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    return d


def test_missing_volume_is_refused(tmp_path):
    res = _run(tmp_path / "not-there")
    assert res.returncode == 1
    assert "does not exist" in res.stderr


def test_container_disk_masquerading_as_a_volume_is_refused(tmp_path):
    """The failure this guard exists for.

    A network volume is a separate filesystem. A directory that merely has the
    right name, on the same device as /, is container disk -- and everything
    built into it dies with the pod. That is only discovered on the NEXT pod,
    after the env is gone, which is the worst possible time.
    """
    vol = tmp_path / "workspace"
    vol.mkdir()
    res = _run(vol)
    assert res.returncode == 1
    assert "CONTAINER DISK" in res.stderr
    assert "ALLOW_CONTAINER_DISK" in res.stderr, "the refusal must say how to override it"


def test_container_disk_override_is_explicit_and_loud(tmp_path, stub_bin):
    vol = tmp_path / "workspace"
    vol.mkdir()
    (vol / "envs" / "behavior").mkdir(parents=True)      # pretend the env is built
    (vol / "BEHAVIOR-1K").mkdir()
    res = _run(vol, env_extra={"ALLOW_CONTAINER_DISK": "1"}, stub_bin=stub_bin)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "proceeding on container disk anyway" in res.stderr


def test_env_and_dataset_are_placed_on_the_volume(tmp_path, stub_bin):
    vol = tmp_path / "workspace"
    vol.mkdir()
    (vol / "envs" / "behavior").mkdir(parents=True)
    (vol / "BEHAVIOR-1K").mkdir()
    res = _run(vol, env_extra={"ALLOW_CONTAINER_DISK": "1"}, stub_bin=stub_bin)
    assert res.returncode == 0, res.stdout + res.stderr

    env_sh = (vol / "env.sh").read_text()
    # CONDA_ENVS_PATH is the mechanism: conda resolves `-n behavior` against it,
    # so upstream's `conda create -n` lands on the volume with no patch upstream.
    assert f'CONDA_ENVS_PATH="{vol}/envs"' in env_sh
    assert f'OMNIGIBSON_DATA_PATH="{vol}/og-data"' in env_sh
    assert (vol / "og-data").is_dir(), (
        "OmniGibson asserts the data path exists at import; it must be created here")


def test_appdata_stays_off_the_volume(tmp_path, stub_bin):
    """OmniGibson's own macros.py warns that APPDATA on a networked filesystem is
    slow and race-prone. It is a cache -- losing it with the pod is correct."""
    vol = tmp_path / "workspace"
    vol.mkdir()
    (vol / "envs" / "behavior").mkdir(parents=True)
    (vol / "BEHAVIOR-1K").mkdir()
    res = _run(vol, env_extra={"ALLOW_CONTAINER_DISK": "1"}, stub_bin=stub_bin)
    assert res.returncode == 0, res.stdout + res.stderr
    env_sh = (vol / "env.sh").read_text()
    assert "OMNIGIBSON_APPDATA_PATH" in env_sh
    assert f'OMNIGIBSON_APPDATA_PATH="{vol}' not in env_sh, (
        "appdata must NOT be on the volume")


def test_existing_env_is_reused_not_rebuilt(tmp_path, stub_bin):
    """The second pod's path. Upstream's setup.sh hard-errors on an existing env
    name, so the script must not call it -- and the A5000 comparison is only
    cheap if this costs seconds."""
    vol = tmp_path / "workspace"
    vol.mkdir()
    (vol / "envs" / "behavior").mkdir(parents=True)
    (vol / "BEHAVIOR-1K").mkdir()
    res = _run(vol, env_extra={"ALLOW_CONTAINER_DISK": "1"}, stub_bin=stub_bin)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "reusing the env on the volume" in res.stdout
    assert "Skipping install" in res.stdout


def test_the_non_relocatable_reasoning_is_recorded():
    """The comment is the point: without it this reads like pointless indirection
    and gets 'simplified' back to `conda create -n` on container disk."""
    text = SETUP.read_text()
    assert "NOT PATH-RELOCATABLE" in text.upper()
    assert "CONDA_ENVS_PATH" in text
    assert "SAME mount point" in text or "same path" in text.lower()
