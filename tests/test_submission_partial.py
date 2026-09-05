"""Partial submissions are the plan, not an accident -- but a shortfall still is.

On a $200 budget we submit only the tasks we trained. Without an explicit
partial mode the count check fires on every single build ("40 rollouts,
expected 2000"), and a warning that always fires is one nobody reads -- which
is precisely how a genuine shortfall gets shipped.

So: declaring the intended task list silences the *intentional* gap and
nothing else. These tests pin that boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submission.build import N_TASKS, PUBLIC_INSTANCES, validate_rollouts  # noqa: E402

TASKS = ["set_up_a_coffee_station_in_your_kitchen", "putting_shoes_on_rack"]
ALL_INSTANCES = sorted(PUBLIC_INSTANCES)


def _write(json_dir: Path, spec: dict[str, list[int]]) -> Path:
    json_dir.mkdir(parents=True, exist_ok=True)
    for task, instances in spec.items():
        for i in instances:
            (json_dir / f"{task}_{i}_0.json").write_text(
                json.dumps({
                    "task": task, "instance_id": i, "rollout_id": 0,
                    "q_score": {"final": 0.3}, "time": {"normalized_time": 0.9},
                })
            )
    return json_dir.parent


def _joined(warnings: list[str]) -> str:
    return " | ".join(warnings)


def test_complete_intentional_partial_does_not_warn_about_the_count(tmp_path):
    root = _write(tmp_path / "json", {t: ALL_INSTANCES for t in TASKS})
    _, warnings = validate_rollouts(root / "json", "public", TASKS)

    text = _joined(warnings)
    assert "rollouts, expected" not in text, f"count warning fired on a complete partial: {text}"
    # The ceiling notice is not a warning about a defect -- it is the number
    # that decides whether the submission is worth building at all.
    assert "PARTIAL submission: 2/100" in text
    assert f"{2 / N_TASKS:.3f}" in text


def test_shortfall_inside_the_intended_set_still_warns(tmp_path):
    root = _write(tmp_path / "json", {
        TASKS[0]: ALL_INSTANCES,
        TASKS[1]: ALL_INSTANCES[:-3],
    })
    _, warnings = validate_rollouts(root / "json", "public", TASKS)

    text = _joined(warnings)
    assert "shortfall WITHIN the intended set" in text
    assert f"{TASKS[1]}: 17/20 instances" in text


def test_intended_task_with_no_rollouts_warns(tmp_path):
    root = _write(tmp_path / "json", {TASKS[0]: ALL_INSTANCES})
    _, warnings = validate_rollouts(root / "json", "public", TASKS)

    text = _joined(warnings)
    assert "NO rollouts at all" in text
    assert TASKS[1] in text


def test_rollouts_outside_the_plan_warn(tmp_path):
    """A task we did not intend to submit means the run did something else."""
    root = _write(tmp_path / "json", {**{t: ALL_INSTANCES for t in TASKS}, "can_meat": [301]})
    _, warnings = validate_rollouts(root / "json", "public", TASKS)

    text = _joined(warnings)
    assert "not in the intended list" in text
    assert "can_meat" in text


def test_without_an_intended_list_the_full_expectation_still_applies(tmp_path):
    """Legacy behaviour is unchanged, and it points at the new flag."""
    root = _write(tmp_path / "json", {t: ALL_INSTANCES for t in TASKS})
    _, warnings = validate_rollouts(root / "json", "public", None)

    text = _joined(warnings)
    assert "expected 2000" in text
    assert "--intended-task" in text


def test_dev_loop_config_is_a_usable_intended_list():
    """--intended-tasks-from reads this file; an empty `tasks:` would silently
    disable partial mode, so pin that it is populated."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/experiments/001-dev-loop.yaml").read_text()
    )
    assert cfg["tasks"], "001-dev-loop.yaml has no tasks; --intended-tasks-from would fail"
    assert set(cfg["tasks"]) == set(TASKS), cfg["tasks"]
