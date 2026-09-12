"""
The frozen dev subset, pinned.

configs/experiments/001-dev-loop.yaml is the measuring stick: both A/B arms run
it, and docs/AB_PROTOCOL.md §3.1 freezes it. Nothing else validated its contents
against the shortlist it claims to be drawn from, so a task rename upstream or a
stray test instance would only surface on a GPU.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONFIG = REPO / "configs" / "experiments" / "001-dev-loop.yaml"
SHORTLIST = REPO / "analysis" / "reward" / "task_shortlist.csv"
# rel_eval_cost carries 3dp here; the shortlist rounds it to 2.
LENGTHS = REPO / "analysis" / "reward" / "full_corpus_lengths.csv"
RESERVE = "outfit_a_basic_toolbox"

# BEHAVIOR-1K v3.9.2: TEST_INSTANCE_IDS = range(301, 341).
FIRST_TEST_INSTANCE = 301


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def shortlist() -> dict[str, dict]:
    with SHORTLIST.open() as f:
        return {r["task"]: r for r in csv.DictReader(f)}


def test_tasks_are_the_two_we_train(cfg):
    assert cfg["tasks"] == [
        "set_up_a_coffee_station_in_your_kitchen",
        "putting_shoes_on_rack",
    ]


def test_tasks_exist_on_the_shortlist(cfg, shortlist):
    """A task id that is not on the shortlist is a typo that costs a GPU session."""
    missing = [t for t in cfg["tasks"] if t not in shortlist]
    assert not missing, f"not on the shortlist: {missing}"


def test_every_task_is_genuinely_graded(cfg, shortlist):
    """The whole A/B design rests on Q being graded here -- see the 2026-09-11
    revision of docs/AB_PROTOCOL.md. D>1 alone is not enough."""
    for task in cfg["tasks"]:
        row = shortlist[task]
        assert int(row["D"]) > 1, f"{task} is binary; the power table assumes graded"
        assert float(row["frac_intermediate"]) > 0.75, (
            f"{task} is graded in name only (frac_intermediate="
            f"{row['frac_intermediate']})")


def test_the_three_d1_control_tasks_are_gone(cfg):
    """Dropped on the $200 budget. They cost training money and add ~nothing to Q."""
    for dropped in ("turning_on_radio", "hanging_pictures", "vacuuming_floors"):
        assert dropped not in cfg["tasks"]


def test_instances_are_training_ids(cfg):
    """Anything >= 301 is a scored test instance; iterating there tunes on the
    leaderboard. harness/launch.py also guards this at run time."""
    assert cfg["mode"] == "train"
    assert cfg["instances"], "an empty instance list silently evaluates nothing"
    off = [i for i in cfg["instances"] if i >= FIRST_TEST_INSTANCE]
    assert not off, f"test instances in a train-mode config: {off}"


def test_reserve_task_is_documented_but_not_run(cfg, shortlist):
    """The reserve is a swap-in for a failed task, not a third task.

    Adding it to `tasks` would change the budget and the frozen list; it must
    stay a comment until a dated protocol revision says otherwise.
    """
    assert RESERVE not in cfg["tasks"], (
        f"{RESERVE} is the reserve -- adding it is a budget change, not a fallback")
    assert RESERVE in CONFIG.read_text(), (
        f"{RESERVE} must stay documented in the config as the reserve")
    # and it has to actually be a viable substitute if we ever need it
    row = shortlist[RESERVE]
    assert row["tier"] == "graded"
    assert int(row["D"]) > 1
    assert float(row["frac_intermediate"]) > 0.75


def test_quoted_figures_match_their_sources(cfg, shortlist):
    """The config's comment table is read by humans making budget calls. Pin it
    to the CSVs so the relabel that changed phi0_mean cannot strand it again.

    rel_eval_cost is quoted at 3dp from full_corpus_lengths.csv; the shortlist
    rounds the same number to 2dp, so check each against its own source.
    """
    text = CONFIG.read_text()
    with LENGTHS.open() as f:
        lengths = {r["task_name"]: r for r in csv.DictReader(f)}

    for task, cost, offset in (
        ("set_up_a_coffee_station_in_your_kitchen", "0.594", "0.165"),
        ("putting_shoes_on_rack", "0.733", "0.000"),
    ):
        assert float(lengths[task]["rel_eval_cost"]) == pytest.approx(
            float(cost), abs=0.001)
        assert float(shortlist[task]["progress_offset_at_reset"]) == pytest.approx(
            float(offset), abs=0.001)
        assert cost in text and offset in text

    # The pre-relabel figures must not reappear as live values.
    assert "phi0_mean" not in text.split("This replaces the pre-relabel")[0]
