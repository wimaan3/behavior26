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


def test_instance_ids_exist_in_the_challenge_dataset(cfg):
    """Every id must be a real shipped instance, not merely < 301.

    resolve_instance_ids passes train ids through unvalidated (evaluator.py
    l.75-76), so a nonexistent id is not caught by the evaluator either -- it
    fails in Evaluator.load_task_instance with FileNotFoundError, on a rented
    GPU, partway through a sweep.

    Verified 2026-09-11 by enumerating 2026-challenge-task-instances.zip: all
    100 tasks ship exactly 300 training instances, but the range is
    task-dependent -- 50 tasks are ids 0-299 and 50 are 1-300. Both of ours are
    0-299, so id 0 is valid here and would NOT be valid on half the corpus.
    """
    lo, hi = 0, 299                      # measured range for both of our tasks
    bad = [i for i in cfg["instances"] if not lo <= i <= hi]
    assert not bad, (
        f"instance ids outside the {lo}-{hi} range both dev-loop tasks ship: {bad}")


def test_instance_count_matches_the_frozen_design(cfg):
    """n is the design, not an accident.

    n=27 was chosen in revision 2026-09-11c: at N=54 the MDE is 0.049 at f=0.20,
    so the design clears its 0.05 target at pessimistic noise and not merely at
    optimistic noise (n=20 gave 0.058). Changing n changes every MDE the protocol
    quotes, so it must go through a dated revision -- this test is what makes a
    silent edit fail.
    """
    instances = cfg["instances"]
    assert len(instances) == 27, (
        f"n={len(instances)}, but the frozen design is n=27. Changing it needs a "
        "dated revision in docs/AB_PROTOCOL.md.")
    assert len(set(instances)) == len(instances), "duplicate instance ids"
    # A duplicate or a gap would quietly change N without changing len().
    assert instances == list(range(min(instances), max(instances) + 1)), (
        "instance list is not contiguous; keep it a block so N is obvious")


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


def test_protocol_quoted_mde_matches_what_power_py_computes(cfg):
    """The doc, the config and the tool must agree on the design's resolution.

    docs/AB_PROTOCOL.md states a resolution limit of ~0.05 and revision
    2026-09-11c quotes 0.049 at f=0.20 for the frozen design. Those numbers are
    load-bearing -- the pre-registered branches key off them -- but they are
    hand-copied into prose, so nothing but this test stops the config changing
    and the doc keeping the old figure.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from analysis.power import load_tasks, mde, pooled_sigma_w

    tasks = load_tasks(SHORTLIST, tier="", names=cfg["tasks"])
    n_units = len(cfg["tasks"]) * len(cfg["instances"])
    assert n_units == 54, f"N={n_units}; the protocol's tables assume N=54"

    got = mde(n_units, pooled_sigma_w(tasks, 0.20), 0.05, cfg["num_rollouts"])
    assert got == pytest.approx(0.049, abs=0.001), (
        f"power.py gives MDE {got:.4f} at f=0.20, but revision 2026-09-11c "
        "quotes 0.049. Update the revision, or the design drifted.")

    # The headline claim in the top-of-document banner.
    assert got <= 0.05, (
        "the protocol says this design resolves dQ >= ~0.05; it no longer does")


def test_the_detection_floor_is_documented_not_left_to_be_derived(cfg):
    """A reader must learn the limit from us.

    Pins that the protocol states the resolution limit, names k=2 as the binding
    constraint, and pre-registers the response to an inconclusive result. These
    are the parts a reader is most likely to need and least likely to derive.
    """
    text = (REPO / "docs" / "AB_PROTOCOL.md").read_text()
    for phrase in (
        "THE DETECTION FLOOR",
        "Resolution limit, stated up front",
        "PRE-REGISTERED",
        "manipulation check",
    ):
        assert phrase in text, f"AB_PROTOCOL.md no longer states: {phrase!r}"
