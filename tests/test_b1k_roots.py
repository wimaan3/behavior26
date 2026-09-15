"""Root validation for one- and multi-task b1k training. Pure Python.

openpi resolves roots asymmetrically (data_loader.create_b1k_dataset):
  one repo_id  -> dataset_root IS the dataset
  a list       -> dataset_root is the PARENT; each dataset at <root>/<repo_id>,
                  loaded by MultiLeRobotDataset

and MultiLeRobotDataset silently DISABLES any feature not present in every
dataset, logging only a warning. For arm B that means one task without a
`progress` column switches progress off for ALL tasks -- the head trains on
nothing and nothing raises. These tests pin the refusals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from b1k_roots import RootError, validate_roots  # noqa: E402


def _ds(path: Path, *, features=("state", "action"), manifest_eps: int | None = None, episodes=10):
    (path / "meta").mkdir(parents=True)
    feats = {f: {"dtype": "float32", "shape": [1]} for f in features}
    (path / "meta" / "info.json").write_text(json.dumps({"total_episodes": episodes, "features": feats}))
    if manifest_eps is not None:
        (path / "meta" / "progress_filter.json").write_text(json.dumps({"episodes_out": manifest_eps}))
    return path


def test_single_task_root_is_the_dataset(tmp_path):
    root = _ds(tmp_path / "coffee")
    plan = validate_roots(root, ["coffee"])
    assert plan.multi is False and plan.dataset_dirs == [root]


def test_multi_task_root_is_the_parent(tmp_path):
    _ds(tmp_path / "coffee"); _ds(tmp_path / "shoes")
    plan = validate_roots(tmp_path, ["coffee", "shoes"])
    assert plan.multi is True
    assert plan.dataset_dirs == [tmp_path / "coffee", tmp_path / "shoes"]


def test_a_missing_task_directory_is_refused(tmp_path):
    _ds(tmp_path / "coffee")
    with pytest.raises(RootError, match="shoes"):
        validate_roots(tmp_path, ["coffee", "shoes"])


def test_progress_missing_from_one_task_is_refused_not_silently_disabled(tmp_path):
    _ds(tmp_path / "coffee", features=("state", "action", "progress"))
    _ds(tmp_path / "shoes", features=("state", "action"))
    with pytest.raises(RootError, match="disable"):
        validate_roots(tmp_path, ["coffee", "shoes"], progress_key="progress")


def test_progress_present_everywhere_passes(tmp_path):
    for t in ("coffee", "shoes"):
        _ds(tmp_path / t, features=("state", "action", "progress"))
    validate_roots(tmp_path, ["coffee", "shoes"], progress_key="progress")


def test_protocol_run_requires_every_task_to_be_filtered_and_compacted(tmp_path):
    _ds(tmp_path / "coffee", manifest_eps=10)
    _ds(tmp_path / "shoes", manifest_eps=None)
    with pytest.raises(RootError, match="progress_filter"):
        validate_roots(tmp_path, ["coffee", "shoes"], protocol=True)


def test_protocol_run_refuses_a_manifest_that_disagrees_with_its_root(tmp_path):
    _ds(tmp_path / "coffee", manifest_eps=9, episodes=10)
    with pytest.raises(RootError, match="9.*10|10.*9"):
        validate_roots(tmp_path / "coffee", ["coffee"], protocol=True)


def test_duplicate_repo_ids_are_refused(tmp_path):
    """Listing a task twice doubles its sampling weight silently."""
    _ds(tmp_path / "coffee")
    with pytest.raises(RootError, match="duplicate"):
        validate_roots(tmp_path, ["coffee", "coffee"])


@pytest.mark.parametrize("script", ["first_batch.py", "compute_norm_stats_b1k.py", "train_b1k_rooted.py"])
def test_every_entry_point_uses_the_shared_root_logic(script):
    """Three copies of root-setting code is how one arm ends up on a different
    data path. Each entry point must go through b1k_roots, not replace
    base_config.dataset_root itself."""
    text = (REPO / "scripts" / script).read_text()
    assert "validate_roots(" in text and "apply_to_config(" in text
    assert "replace(cfg.data.base_config, dataset_root" not in text, f"{script} sets the root inline"
