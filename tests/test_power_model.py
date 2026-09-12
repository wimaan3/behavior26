"""
The noise model behind the A/B design, pinned.

analysis/power.py decides how we spend the eval budget. Its sigma_w
parameterisation changed when the Jetson's relabel showed the tasks we actually
run are graded (D=6, D=10), not binary -- so the properties that justify the
design are asserted here rather than left to a docstring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.power import (  # noqa: E402
    credit_step, load_tasks, mde, pooled_sigma_w, sigma_w_for,
)

SHORTLIST = REPO / "analysis" / "reward" / "task_shortlist.csv"


def _task(name: str) -> dict:
    rows = load_tasks(SHORTLIST, tier="", names=[name])
    assert rows, f"{name} not on the shortlist"
    return rows[0]


# --------------------------------------------------------------------------------------
# the noise model
# --------------------------------------------------------------------------------------
def test_binary_case_reduces_to_the_old_formula():
    """At D=1 the graded model must give back sigma_w^2 = 0.25f exactly.

    The D=1 tier is still on the shortlist and §0 of the A/B protocol still
    governs it, so the generalisation has to be backward compatible.
    """
    radio = _task("turning_on_radio")
    assert radio["D"] == 1
    assert credit_step(radio) == pytest.approx(1.0)
    for f in (0.05, 0.10, 0.20, 0.40):
        assert sigma_w_for(radio, f) == pytest.approx(math.sqrt(0.25 * f))


def test_grading_shrinks_noise_by_root_d():
    """A one-unit wobble moves Q by 1/D, so sigma_w falls as sqrt(D) at fixed f."""
    radio = _task("turning_on_radio")                          # D=1
    coffee = _task("set_up_a_coffee_station_in_your_kitchen")  # D=6
    shoes = _task("putting_shoes_on_rack")                     # D=10
    assert (coffee["D"], shoes["D"]) == (6, 10)

    f = 0.20
    binary = sigma_w_for(radio, f)
    assert sigma_w_for(coffee, f) / binary == pytest.approx(1 / math.sqrt(6), rel=0.05)
    assert sigma_w_for(shoes, f) / binary == pytest.approx(1 / math.sqrt(10), rel=0.05)
    assert sigma_w_for(shoes, f) < sigma_w_for(coffee, f) < binary


def test_measured_step_matches_the_one_over_d_idealisation():
    """max_step_frac is measured, 1/D is assumed. They must agree, or D is not
    telling us what we think about how credit is awarded."""
    for name in ("set_up_a_coffee_station_in_your_kitchen", "putting_shoes_on_rack"):
        t = _task(name)
        assert t["max_step_frac"] == pytest.approx(1 / t["D"], rel=0.05)


def test_both_dev_loop_tasks_are_genuinely_graded():
    """D > 1 is not enough -- the episode has to spend time at partial credit.

    cook_bacon is D=7 but the counter-example matters: a task can be nominally
    graded and behave as binary. Guard the two we actually run.
    """
    for name in ("set_up_a_coffee_station_in_your_kitchen", "putting_shoes_on_rack"):
        t = _task(name)
        assert t["D"] > 1
        assert t["frac_intermediate"] > 0.75, (
            f"{name} is graded in name only; the binary noise model would apply"
        )


def test_pooled_sigma_uses_mean_of_variances():
    """Units enter the mean with equal weight, so pool sigma_w^2, not sigma_w."""
    coffee = _task("set_up_a_coffee_station_in_your_kitchen")
    shoes = _task("putting_shoes_on_rack")
    f = 0.20
    expected = math.sqrt(
        (sigma_w_for(coffee, f) ** 2 + sigma_w_for(shoes, f) ** 2) / 2)
    assert pooled_sigma_w([coffee, shoes], f) == pytest.approx(expected)
    # and it must sit between the two, not outside them
    assert sigma_w_for(shoes, f) < expected < sigma_w_for(coffee, f)


# --------------------------------------------------------------------------------------
# what the model implies for the design
# --------------------------------------------------------------------------------------
def test_graded_tasks_beat_binary_at_equal_n():
    """The whole reason the budget conversation changed."""
    f, n, m = 0.20, 40, 1
    binary = mde(n, sigma_w_for(_task("turning_on_radio"), f), 0.05, m)
    graded = mde(n, pooled_sigma_w(
        [_task("set_up_a_coffee_station_in_your_kitchen"),
         _task("putting_shoes_on_rack")], f), 0.05, m)
    assert graded < binary / 2


def test_sigma_b_is_a_floor_that_seeds_cannot_cross():
    """Seeds shrink 2*sigma_w^2/m and nothing else, so the MDE converges to a
    floor set by sigma_b that no number of seeds crosses."""
    tasks = [_task("set_up_a_coffee_station_in_your_kitchen"),
             _task("putting_shoes_on_rack")]
    sigma_b, n, f = 0.05, 20, 0.20
    sigma = pooled_sigma_w(tasks, f)
    floor = mde(n, 0.0, sigma_b, 1)
    assert mde(n, sigma, sigma_b, 1) > mde(n, sigma, sigma_b, 3) > floor
    assert mde(n, sigma, sigma_b, 1000) == pytest.approx(floor, rel=0.01)


def test_grading_raises_sigma_b_share_of_the_variance():
    """Why the design conversation changed shape, stated as a number.

    Under binary Q, sigma_w swamped sigma_b and seeds were nearly the only
    lever -- which is what made the §2 noise-floor measurement decisive. Graded
    Q cuts sigma_w by ~sqrt(D), so effect heterogeneity becomes a real share of
    the variance and the most seeds can ever buy shrinks with it.
    """
    sigma_b, f = 0.05, 0.20
    binary = sigma_w_for(_task("turning_on_radio"), f)
    graded = pooled_sigma_w([_task("set_up_a_coffee_station_in_your_kitchen"),
                             _task("putting_shoes_on_rack")], f)

    def sigma_b_share(sigma_w: float) -> float:
        return sigma_b ** 2 / (2 * sigma_w ** 2 + sigma_b ** 2)

    assert sigma_b_share(binary) < 0.05      # binary: sigma_b is a rounding error
    assert sigma_b_share(graded) > 0.15      # graded: it is a real term


def test_instances_beat_seeds_at_equal_rollout_count():
    """Doubling n and doubling m cost the same rollouts; n also cuts the sigma_b
    term, which m never touches."""
    tasks = [_task("set_up_a_coffee_station_in_your_kitchen"),
             _task("putting_shoes_on_rack")]
    sigma_b, f = 0.05, 0.20
    sigma = pooled_sigma_w(tasks, f)
    more_instances = mde(40, sigma, sigma_b, 1)   # 2 tasks x 20 instances, 1 seed
    more_seeds = mde(20, sigma, sigma_b, 2)       # 2 tasks x 10 instances, 2 seeds
    assert more_instances < more_seeds


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def test_pre_relabel_shortlist_still_loads(tmp_path):
    """The old schema has no max_step_frac. Fall back to 1/D rather than crash."""
    csv_path = tmp_path / "old.csv"
    csv_path.write_text(
        "rank,tier,task,task_index,D,eval_timeout_frames,rel_eval_cost\n"
        "1,primary,turning_on_radio,0,1,3224,0.2\n"
        "2,primary,cook_bacon,46,7,11519,0.73\n"
    )
    rows = load_tasks(csv_path, tier="primary")
    assert [r["task"] for r in rows] == ["turning_on_radio", "cook_bacon"]
    assert rows[0]["max_step_frac"] is None
    assert credit_step(rows[0]) == pytest.approx(1.0)
    assert credit_step(rows[1]) == pytest.approx(1 / 7)
    assert sigma_w_for(rows[0], 0.2) == pytest.approx(math.sqrt(0.25 * 0.2))


def test_named_tasks_override_tier():
    """--tasks-from must reach the graded tier even though --tier defaults to primary."""
    rows = load_tasks(SHORTLIST, tier="primary",
                      names=["putting_shoes_on_rack",
                             "set_up_a_coffee_station_in_your_kitchen"])
    assert len(rows) == 2
    assert {r["task"] for r in rows} == {
        "putting_shoes_on_rack", "set_up_a_coffee_station_in_your_kitchen"}
    # cheapest-first
    assert rows[0]["task"] == "set_up_a_coffee_station_in_your_kitchen"


def test_unknown_task_name_is_an_error():
    with pytest.raises(SystemExit):
        load_tasks(SHORTLIST, tier="", names=["no_such_task"])
