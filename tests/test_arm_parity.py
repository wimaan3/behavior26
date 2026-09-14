"""Both arms must train on the SAME filtered root -- as a mechanism, not a rule.

AB_PROTOCOL 3.6 says it, but saying it is what we had. The natural mistake is
arm A pointed at the pristine slice (200 episodes) while arm B trains on the
merged one (199): dQ would then measure the progress head PLUS 0.5% more data,
and nothing in the run would say so.

`meta/progress_filter.json` already existed as evidence. These tests wire it into
a check that refuses to start.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train_cloud.sh"


def _root(tmp_path: Path, name: str, *, episodes: int, manifest_episodes: int | None) -> Path:
    """A minimal root: info.json with a count, and optionally a filter manifest."""
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        {"codebase_version": "v3.0", "fps": 30, "total_episodes": episodes, "total_frames": episodes * 10}))
    if manifest_episodes is not None:
        (root / "meta" / "progress_filter.json").write_text(json.dumps(
            {"episodes_out": manifest_episodes, "compacted": True, "n_dropped_episodes": 1}))
    return root


def _run(dataset_root: Path, *, arm: str | None, extra: dict | None = None):
    # OPENPI_ROOT must exist: the script's disk check runs against it.
    openpi = dataset_root.parent / "openpi"
    openpi.mkdir(exist_ok=True)
    env = dict(os.environ, DATASET_ROOT=str(dataset_root), DRY_RUN="1", SKIP_BASHRC="1",
               OPENPI_ROOT=str(openpi))
    if arm:
        env["ARM"] = arm
    env.update(extra or {})
    for leak in ("PROGRESS_KEY",):
        env.pop(leak, None)
    return subprocess.run(["bash", str(TRAIN), "--dry-run"], capture_output=True, text=True, env=env)


def test_a_protocol_run_refuses_an_unfiltered_root(tmp_path):
    """Arm A on the pristine slice is the mistake this exists to prevent."""
    pristine = _root(tmp_path, "slice", episodes=200, manifest_episodes=None)
    res = _run(pristine, arm="A")
    assert res.returncode != 0, res.stdout
    assert "progress_filter.json" in (res.stdout + res.stderr)


def test_a_protocol_run_refuses_a_root_whose_count_disagrees_with_its_manifest(tmp_path):
    """Catches a root that was dropped but never compacted, and a manifest copied
    next to the wrong data."""
    mismatched = _root(tmp_path, "merged", episodes=200, manifest_episodes=199)
    res = _run(mismatched, arm="B")
    assert res.returncode != 0, res.stdout
    out = res.stdout + res.stderr
    assert "199" in out and "200" in out


def test_both_arms_accept_the_same_filtered_root(tmp_path):
    """The positive case: one root, two arms, and the resolved path is identical."""
    good = _root(tmp_path, "merged", episodes=199, manifest_episodes=199)
    a = _run(good, arm="A")
    b = _run(good, arm="B", extra={"PROGRESS_KEY": "progress", "CONFIG": "pi05_b1k_frozen_vlm_progress"})
    assert a.returncode == 0, a.stdout + a.stderr
    assert b.returncode == 0, b.stdout + b.stderr
    def resolved(res):
        return [l for l in res.stdout.splitlines() if "dataset:" in l or "arm root" in l]
    assert resolved(a), a.stdout
    assert str(good) in a.stdout and str(good) in b.stdout


def test_a_non_protocol_run_still_works_but_says_so(tmp_path):
    """Rungs 1-4 measure throughput on a raw slice and must not be blocked -- but
    an unmarked run should not look like a protocol run either."""
    pristine = _root(tmp_path, "slice", episodes=200, manifest_episodes=None)
    res = _run(pristine, arm=None)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "not a protocol run" in (res.stdout + res.stderr).lower()


def test_the_guard_is_documented_where_the_rule_is():
    text = (REPO / "docs" / "AB_PROTOCOL.md").read_text()
    assert "ARM=A" in text and "progress_filter.json" in text


def test_validate_slice_checks_compaction_as_a_first_class_check():
    """4b caught this speculatively; that is the argument for promoting it."""
    text = (REPO / "scripts" / "validate_slice.py").read_text()
    assert "5 merged root is compacted" in text
    assert "5b filter manifest agrees with the root" in text
    # structural, not dependent on LeRobot raising a confusing 401
    seg = text.split("5 -- STRUCTURAL")[1][:2000]
    assert "episode_index not dense" in seg and "index not dense" in seg
