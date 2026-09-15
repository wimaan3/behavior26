"""Tests for analysis.reward.labels, run against REAL LeRobot parquet.

Requires a local LeRobot pull containing at least chunks 000, 008, 011, 046, 069.
Point BEHAVIOR_DATA_ROOT at it (default ~/behavior-data). Tests skip if absent.

The five tasks are chosen deliberately:
  task 0   turning_on_radio                    D=1,  clean, has rollback negatives
  task 8   rearranging_kitchen_furniture       no reward signal at all -> must refuse
  task 11  putting_dishes_away_after_cleaning  D=14, phi0=0.93 under-fires -> must refuse
  task 46  cook_bacon                          D=7,  rich structure, 2 over-firing episodes
  task 69  vacuuming_floors                    D=1,  clean
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from analysis.reward import labels

DATA_ROOT = pathlib.Path(os.environ.get("BEHAVIOR_DATA_ROOT", pathlib.Path.home() / "behavior-data"))
REWARD_MAP = pathlib.Path(__file__).resolve().parents[1] / "behavior1k_reward_map.json"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "info.json").exists(),
    reason=f"no local LeRobot pull at {DATA_ROOT}",
)


@pytest.fixture(scope="module")
def rmap():
    return labels.load_reward_map(REWARD_MAP)


@pytest.fixture(scope="module")
def ds():
    return labels.Dataset(DATA_ROOT)


# --------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------

def test_repair_zeroes_a_negative_that_would_drive_the_running_sum_below_zero():
    r = np.array([0.0, -1.0, 0.0, 1.0, 0.0])
    repaired, dropped = labels.repair_negatives(r)
    assert dropped == 1
    assert repaired.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_repair_keeps_a_negative_that_stays_at_or_above_the_floor():
    r = np.array([1.0, 1.0, -1.0])
    repaired, dropped = labels.repair_negatives(r)
    assert dropped == 0
    assert repaired.tolist() == [1.0, 1.0, -1.0]


def test_repair_on_real_rollback_episode_restores_full_headroom(ds):
    """turning_on_radio episode 0: reward is -1 at t=1246 then +1 at t=1364.

    Unrepaired the trace nets to zero and phi0 would be 1.0 -- claiming the task
    is already complete at t=0. The repair must drop exactly the one negative.
    """
    r = ds.episode_rewards(task_index=0, episode_index=0)
    assert (r < 0).sum() == 1, "fixture episode no longer contains a rollback negative"

    repaired, dropped = labels.repair_negatives(r)
    assert dropped == 1

    ep = labels.episode_labels(r, D=1)
    assert ep.progress[0] == pytest.approx(0.0)
    assert ep.progress[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# phi / progress
# --------------------------------------------------------------------------

def test_progress_ends_at_one_and_is_bounded(ds):
    r = ds.episode_rewards(task_index=69, episode_index=13801)
    ep = labels.episode_labels(r, D=1)
    assert ep.progress[-1] == pytest.approx(1.0)
    assert ep.progress.min() >= 0.0 and ep.progress.max() <= 1.0


def test_progress_length_matches_episode_length(ds):
    r = ds.episode_rewards(task_index=0, episode_index=0)
    ep = labels.episode_labels(r, D=1)
    assert len(ep.progress) == len(r) == 1956


def test_satisfied_count_is_integral_on_a_multi_unit_task(ds):
    """cook_bacon has D=7; every satisfied count must land on an integer."""
    kept = 0
    for episode_index in ds.episode_indices(task_index=46)[:20]:
        r = ds.episode_rewards(task_index=46, episode_index=episode_index)
        ep = labels.episode_labels(r, D=7)
        if not ep.valid:
            continue
        kept += 1
        residual = np.abs(ep.satisfied_count - np.round(ep.satisfied_count))
        assert residual.max() < 1e-4
    assert kept > 0


def test_satisfied_count_is_progress_times_D(ds):
    r = ds.episode_rewards(task_index=46, episode_index=9200)
    ep = labels.episode_labels(r, D=7)
    np.testing.assert_allclose(ep.satisfied_count, ep.progress * 7, atol=1e-5)


# --------------------------------------------------------------------------
# validation gates
# --------------------------------------------------------------------------

def test_over_firing_episode_is_dropped_as_over_credited(ds):
    """cook_bacon episode 9203 credits 8 of its 7 units.

    The terminal anchor saw this as phi0 = -1/7, out of range. The init anchor sees it
    for what it is -- a trace that credits more of the goal than the goal contains,
    which is the rollback-replay signature -- and names the gate accordingly.
    """
    r = ds.episode_rewards(task_index=46, episode_index=9203)
    ep = labels.episode_labels(r, D=7)
    assert not ep.valid
    assert ep.drop_reason == "over_credited"
    assert ep.peak > 1.0


def test_clean_episode_passes_every_gate(ds):
    r = ds.episode_rewards(task_index=0, episode_index=1)
    ep = labels.episode_labels(r, D=1, terminated_last=True, truncated_any=False)
    assert ep.valid and ep.drop_reason is None


def test_episode_that_never_terminates_is_dropped(ds):
    r = ds.episode_rewards(task_index=0, episode_index=1)
    ep = labels.episode_labels(r, D=1, terminated_last=False, truncated_any=False)
    assert not ep.valid
    assert ep.drop_reason == "not_terminated"


# --------------------------------------------------------------------------
# D
# --------------------------------------------------------------------------

def test_D_measured_from_data_matches_the_reward_map(ds, rmap):
    for task_index, task_name in [(0, "turning_on_radio"), (46, "cook_bacon"),
                                  (11, "putting_dishes_away_after_cleaning")]:
        measured = ds.measure_D(task_index)
        assert measured == rmap[task_name]["D"], task_name


def test_measure_D_returns_none_when_the_task_emits_no_reward(ds):
    assert ds.measure_D(8) is None


def test_all_magnitudes_are_integer_multiples_of_one_over_D(ds):
    mags = ds.reward_magnitudes(task_index=46)
    D = 7
    for m in mags:
        assert abs(m * D - round(m * D)) < 1e-4, m


# --------------------------------------------------------------------------
# refusal
# --------------------------------------------------------------------------

def test_refuses_a_task_with_no_reward_signal(rmap):
    with pytest.raises(labels.TaskRefused) as exc:
        labels.check_task_usable("rearranging_kitchen_furniture", rmap)
    assert "no reward signal" in str(exc.value).lower()


def test_refuses_the_under_instrumented_reference_task(rmap):
    """putting_dishes_away_after_cleaning: D=14 but a successful demo fires once."""
    with pytest.raises(labels.TaskRefused) as exc:
        labels.check_task_usable("putting_dishes_away_after_cleaning", rmap)
    msg = str(exc.value)
    assert "PROVEN_incomplete" in msg


def test_refuses_an_unknown_task(rmap):
    with pytest.raises(labels.TaskRefused):
        labels.check_task_usable("no_such_task_exists", rmap)


def test_accepts_a_clean_task(rmap):
    labels.check_task_usable("turning_on_radio", rmap)   # must not raise


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_build_task_labels_writes_parquet_aligned_to_frame_index(tmp_path, ds, rmap):
    man = labels.build_task_labels("turning_on_radio", ds, rmap, tmp_path, max_episodes=5)
    out = tmp_path / "turning_on_radio" / "labels.parquet"
    assert out.exists()

    t = pq.read_table(out).to_pandas()
    assert set(["index", "episode_index", "frame_index", "task_index",
                "progress", "satisfied_count"]).issubset(t.columns)

    for ep, grp in t.groupby("episode_index"):
        grp = grp.sort_values("frame_index")
        assert grp.frame_index.tolist() == list(range(len(grp)))
        assert grp.progress.iloc[-1] == pytest.approx(1.0)
        assert grp.progress.is_monotonic_increasing
    assert man["episodes_used"] == 5


def test_manifest_records_D_counts_drop_reasons_and_the_reset_offset(tmp_path, ds, rmap):
    man = labels.build_task_labels("cook_bacon", ds, rmap, tmp_path, max_episodes=10)
    assert man["task"] == "cook_bacon"
    assert man["D"] == 7
    assert man["episodes_used"] + man["episodes_dropped"] == 10
    assert man["episodes_dropped"] >= 1
    assert "over_credited" in man["drop_reasons"]
    # one of cook_bacon's seven literals -- `not open(refrigerator)` -- is satisfied at
    # reset, so progress starts at 1/7 in every episode that begins with the door shut.
    assert man["initially_true_literals"] == 1
    assert man["census_alignment"] == "aligned"
    assert 0.0 <= man["mean_phi0"] <= 1.0 / 7 + 1e-9
    assert man["mean_never_credited_units"] == pytest.approx(0.0, abs=1e-6)
    assert man["demo_completion_rate"] == pytest.approx(1.0)


def test_build_refuses_and_writes_no_labels_for_the_defective_task(tmp_path, ds, rmap):
    with pytest.raises(labels.TaskRefused):
        labels.build_task_labels("putting_dishes_away_after_cleaning", ds, rmap, tmp_path)
    assert not (tmp_path / "putting_dishes_away_after_cleaning" / "labels.parquet").exists()


def test_labels_index_column_joins_back_to_the_source_parquet(tmp_path, ds, rmap):
    """The emitted `index` must be the LeRobot dataset-global row index."""
    labels.build_task_labels("turning_on_radio", ds, rmap, tmp_path, max_episodes=3)
    lab = pq.read_table(tmp_path / "turning_on_radio" / "labels.parquet").to_pandas()

    src = ds.read_columns(task_index=0, columns=["index", "episode_index", "frame_index"])
    src = src[src.episode_index.isin(lab.episode_index.unique())]
    merged = lab.merge(src, on="index", suffixes=("", "_src"))
    assert len(merged) == len(lab)
    assert (merged.frame_index == merged.frame_index_src).all()
    assert (merged.episode_index == merged.episode_index_src).all()


# --------------------------------------------------------------------------
# regression: reproduce the committed corpus sweep from raw parquet
# --------------------------------------------------------------------------

EPISODE_STATS = pathlib.Path(__file__).resolve().parents[1] / "behavior1k_episode_stats.csv"


def _historical(task_index):
    import csv
    with open(EPISODE_STATS) as fh:
        # The file carries a '#' preamble marking it terminal-anchored and superseded;
        # csv.DictReader would otherwise take the first comment line as the header.
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {int(r["ep"]): r for r in rows if int(r["task"]) == task_index}


@pytest.mark.parametrize("task_index,task_name,D", [
    (0, "turning_on_radio", 1),
    (69, "vacuuming_floors", 1),
    (46, "cook_bacon", 7),
])
def test_pipeline_reproduces_the_committed_sweep(ds, task_index, task_name, D):
    """phi0, repair count and validity must match behavior1k_episode_stats.csv exactly.

    That CSV is the 5,677-episode measurement the whole recipe rests on. If this
    drifts, either the pipeline changed or the dataset was re-patched underneath us.

    It was measured with the terminal anchor, so the pin names that anchor explicitly
    rather than tracking whatever the default happens to be. The two anchors agree
    exactly on turning_on_radio and vacuuming_floors -- on a task with nothing true at
    reset and demos that all finish, the init anchor reduces to the old formula. They
    part company on cook_bacon, which has both.
    """
    hist = _historical(task_index)
    assert hist, f"no rows for task {task_index} in {EPISODE_STATS}"

    frame = ds.rewards_frame(task_index)
    checked = 0
    for ep, row in sorted(hist.items()):
        e = frame[frame.episode_index == ep]
        assert len(e) == int(row["T"]), f"ep {ep} length drift"
        lab = labels.episode_labels(
            e["next.reward"].to_numpy(np.float64), D,
            terminated_last=bool(e["next.terminated"].iloc[-1]),
            truncated_any=bool(e["next.truncated"].any()),
            anchor="terminal",
        )
        assert lab.phi0 == pytest.approx(float(row["phi0"]), abs=1e-4), f"ep {ep} phi0"
        assert lab.n_dropped_rewards == int(row["dropped"]), f"ep {ep} repair count"
        assert lab.valid == (row["valid"] == "True"), f"ep {ep} validity"
        checked += 1
    assert checked == len(hist)


def test_episode_with_zero_headroom_is_dropped(ds):
    """An episode that terminates having fired no reward at all must not be labelled.

    phi0 == 1 means progress is the constant 1.0 for every frame: the label claims a
    freshly-reset scene is already complete. It passes range and integrality (1 is in
    [0,1] and 1*D is integral), so only an explicit headroom gate catches it. One of
    the 200 vacuuming_floors episodes is exactly this -- invisible in the 105-episode
    sample the committed corpus stats were measured on.
    """
    r = np.zeros(500)
    ep = labels.episode_labels(r, D=1, terminated_last=True)
    assert not ep.valid
    assert ep.drop_reason == "no_headroom"


def test_zero_headroom_episode_is_excluded_from_real_task_labels(tmp_path, ds, rmap):
    man = labels.build_task_labels("vacuuming_floors", ds, rmap, tmp_path)
    assert man["drop_reasons"].get("no_headroom") == 1
    assert man["mean_phi0"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# the refusal gate must not read stale sample-scope phi0
# --------------------------------------------------------------------------

def test_reward_map_prefers_full_corpus_phi0_over_the_sample():
    """installing_a_fax_machine reads phi0=0 on its 62-episode shard sample and
    0.1075 across all 200. The refusal gate must see the full-corpus number."""
    rmap = labels.load_reward_map(REWARD_MAP)
    rec = rmap["installing_a_fax_machine"]
    assert rec["phi0_mean"] == pytest.approx(0.1075, abs=1e-3)
    assert rec["measurement_scope"] == "full-200ep"


def test_the_task_that_only_under_fired_at_full_scale_is_now_accepted():
    """installing_a_fax_machine is not an instrumentation defect -- its demos truncate.

    Terminal-anchored phi0 read 0.1075 across 200 episodes and the task was refused as
    under-instrumented. Init-anchored, 78.5% of its episodes reach progress 1.0 and the
    rest stop one unit short; nothing is wrong with the reward. The gate must accept it
    and the status must say which of the two it is.
    """
    rmap = labels.load_reward_map(REWARD_MAP)
    rec = labels.check_task_usable("installing_a_fax_machine", rmap)   # must not raise
    assert rec["demo_completion_rate"] > 0.5
    status, _ = labels.denominator_status(
        "installing_a_fax_machine", int(rec["D"]),
        float(rec["never_credited_mean"]), float(rec["demo_completion_rate"]))
    assert status == "ENDS_SHORT"


def test_still_refuses_when_no_demo_in_the_task_ever_completes():
    """The gate that replaced phi0 must catch the reference defect on its own.

    putting_dishes_away_after_cleaning is also flagged `PROVEN_incomplete` in the map,
    which short-circuits first, so the flag is cleared here to leave the completion
    gate as the only thing standing between this task and a label. Its measured
    numbers are untouched.
    """
    rmap = labels.load_reward_map(REWARD_MAP)
    rec = dict(rmap["putting_dishes_away_after_cleaning"], reward_instrumentation="ok")
    assert rec["demo_completion_rate"] == 0.0
    assert rec["never_credited_mean"] > 12
    with pytest.raises(labels.TaskRefused) as exc:
        labels.check_task_usable("putting_dishes_away_after_cleaning",
                                 {"putting_dishes_away_after_cleaning": rec})
    msg = str(exc.value)
    assert "no demo reaches the whole goal" in msg and "14" in msg


def test_sample_scope_is_kept_for_tasks_with_no_full_measurement(tmp_path):
    """The overlay must apply only where a full measurement exists.

    Every task now has one, so no real task exercises the fallback -- it is driven
    here with a remeasure file that omits a task on purpose. Asserting against a task
    that merely happens not to be re-measured yet is a test that decays silently.
    """
    full = pd.read_csv(pathlib.Path(REWARD_MAP).parent / "full_corpus_remeasure.csv")
    partial = tmp_path / "partial_remeasure.csv"
    full[full.task_name != "cook_a_frozen_pie"].to_csv(partial, index=False)

    rmap = labels.load_reward_map(REWARD_MAP, remeasure=partial)
    assert rmap["cook_a_frozen_pie"]["measurement_scope"] == "sample-28pct"
    assert rmap["turning_on_radio"]["measurement_scope"] == "full-200ep"


def test_every_task_now_has_a_full_corpus_measurement():
    """Task selection reads the overlay, so a gap in it is a silent fallback to the
    28.4% sample -- the exact blindness that made installing_a_fax_machine rank 9."""
    rmap = labels.load_reward_map(REWARD_MAP)
    sampled = sorted(k for k, v in rmap.items()
                     if v.get("measurement_scope") != "full-200ep")
    assert sampled == [], f"still sample-scope: {sampled}"


def test_no_signal_tasks_carry_full_scope_too():
    """Their verdict does not change at full scale, but the strength of it does."""
    rmap = labels.load_reward_map(REWARD_MAP)
    for task in ("rearranging_kitchen_furniture", "storing_food"):
        rec = rmap[task]
        assert rec["D"] is None and rec["measurement_scope"] == "full-200ep"
        assert rec["episodes"] == 200


# --------------------------------------------------------------------------
# collapsed denominators (warning 3)
# --------------------------------------------------------------------------

def test_manifest_records_the_denominator_status(tmp_path, ds, rmap):
    man = labels.build_task_labels("make_microwave_popcorn", ds, rmap, tmp_path, max_episodes=3)
    assert man["denominator_status"] == "COLLAPSED"
    assert man["must_flip_predicates"] == 2 and man["D"] == 1

    clean = labels.build_task_labels("turning_on_radio", ds, rmap, tmp_path, max_episodes=3)
    assert clean["denominator_status"] == "CONSISTENT"


def test_collapsed_denominator_is_refused_when_asked(tmp_path, ds, rmap):
    """phi0 is blind to a D collapsed below the goal, so it needs its own switch."""
    with pytest.raises(labels.TaskRefused) as exc:
        labels.build_task_labels("make_microwave_popcorn", ds, rmap, tmp_path,
                                 max_episodes=3, refuse_collapsed=True)
    assert "COLLAPSED" in str(exc.value)
    assert not (tmp_path / "make_microwave_popcorn" / "labels.parquet").exists()


# --------------------------------------------------------------------------
# the init anchor
# --------------------------------------------------------------------------

def test_init_anchor_matches_the_old_formula_when_nothing_is_true_at_reset():
    """With s0 = 0 and a demo that finishes, the two anchors are the same curve.

    This is why 12 of the 14 CONSISTENT tasks are unaffected by the change: their goal
    is entirely false at reset and their demos all reach it.
    """
    r = np.zeros(10)
    r[3] = r[7] = 0.5
    init = labels.episode_labels(r, D=2)
    term = labels.episode_labels(r, D=2, anchor="terminal")
    np.testing.assert_allclose(init.progress, term.progress, atol=1e-9)
    assert init.phi0 == pytest.approx(0.0) and init.final == pytest.approx(1.0)


def test_short_demo_keeps_the_progress_it_actually_reached():
    """A demo that stops a unit short must not be back-shifted into starting ahead.

    Terminal anchor: 3 of 4 units credited -> the whole curve is lifted by 1/4 and the
    episode is reported as 25% complete before the robot moves. That is the reading
    that put 39 tasks in the UNDER_FIRES band. Init anchor: starts at 0, ends at 0.75.
    """
    r = np.zeros(20)
    r[[4, 9, 14]] = 0.25
    init = labels.episode_labels(r, D=4)
    term = labels.episode_labels(r, D=4, anchor="terminal")
    assert init.phi0 == pytest.approx(0.0)
    assert init.final == pytest.approx(0.75)
    assert init.never_credited_units == pytest.approx(1.0)
    assert init.lost_at_end_units == pytest.approx(0.0)
    assert term.phi0 == pytest.approx(0.25)
    assert term.final == pytest.approx(1.0)


def test_a_literal_true_at_reset_puts_the_curve_above_zero_at_t0():
    """s0=1 of D=7: the trace debits for breaking it, so the debit must survive repair."""
    r = np.zeros(10)
    r[2] = -1 / 7        # the initially-true literal is broken
    r[5] = 1 / 7         # and restored
    r[8] = 6 / 7         # the six real units land
    ep = labels.episode_labels(r, D=7, s0_units=1)
    assert ep.phi0 == pytest.approx(1 / 7)
    assert ep.progress[3] == pytest.approx(0.0)      # broken
    assert ep.final == pytest.approx(1.0)
    assert ep.n_dropped_rewards == 0                 # nothing repaired away
    assert not ep.monotone and ep.max_dip_units == pytest.approx(1.0)


def test_the_census_prior_is_capped_by_what_the_trace_credits():
    """A prior of 1 unit cannot stand when the trace credits all D of them.

    44 of 200 cook_bacon episodes earn credit for CLOSING the refrigerator with no
    preceding debit for opening it, so in those episodes the door was open at reset and
    the census's `not open(...) initially_true` is simply false. Uncapped, s0=1 makes
    the episode credit 8 of 7 units and the whole episode is thrown away.
    """
    r = np.zeros(10)
    r[1] = 1 / 7         # closed, with no debit before it
    r[3] = -1 / 7
    r[5] = 1 / 7
    r[8] = 6 / 7
    ep = labels.episode_labels(r, D=7, s0_units=1)
    assert ep.valid, ep.drop_reason
    assert ep.phi0 == pytest.approx(0.0)             # prior capped away
    assert ep.peak == pytest.approx(1.0)
    assert ep.final == pytest.approx(1.0)


def test_unknown_init_literals_do_not_raise_the_anchor(ds):
    """`initially_true: None` must not be counted, or a rollback debit becomes an s0.

    On a D=1 task that reading pins progress at the constant 1.0 and the episode is
    dropped for having no headroom; it cost 104 of 200 re_shelving_library_books
    episodes while the bound counted unknowns.
    """
    s0, n_unknown, alignment = labels.initial_satisfied_bounds("chop_an_onion", D=4)
    assert alignment == "aligned"
    assert s0 == 0 and n_unknown == 2

    r = np.zeros(500)
    r[100] = -1.0        # rollback debit
    r[200] = 1.0
    ep = labels.episode_labels(r, D=1, s0_units=0)
    assert ep.valid and ep.phi0 == pytest.approx(0.0)
    assert ep.n_dropped_rewards == 1


def test_initial_satisfied_bounds_reports_ambiguity_when_D_is_not_the_literal_count():
    """thawing_frozen_food has 9 goal literals and D=7: no literal maps to a unit."""
    s0, n_unknown, alignment = labels.initial_satisfied_bounds("thawing_frozen_food", D=7)
    assert alignment == "ambiguous" and s0 == 0

    s0, n_unknown, alignment = labels.initial_satisfied_bounds("wash_dog_toys", D=6)
    assert alignment == "aligned" and s0 == 2 and n_unknown == 0


# --------------------------------------------------------------------------
# progress structure
# --------------------------------------------------------------------------

def test_progress_structure_separates_a_step_function_from_a_staircase():
    """D says how many units. It does not say whether they arrive one at a time."""
    step = np.concatenate([np.zeros(90), np.ones(10)])          # D=7, all at once
    stair = np.repeat(np.arange(5) / 4.0, 20)                   # D=4, evenly spread

    a = labels.progress_structure(step, D=7)
    b = labels.progress_structure(stair, D=4)

    assert a["frac_intermediate"] == pytest.approx(0.0)
    assert a["max_step_frac"] == pytest.approx(1.0)
    assert a["n_levels"] == 2

    assert b["frac_intermediate"] > 0.5
    assert b["max_step_frac"] == pytest.approx(0.25)
    assert b["n_levels"] == 5


def test_cook_bacon_lands_six_of_its_seven_units_in_one_frame(ds):
    """The reason D is not a proxy for gradient, measured on the real task.

    Every cook_bacon episode has exactly one +6/7 reward sample, and nothing fires
    after it. A progress label built from this is a step function with a 1/7 bump.
    """
    for episode_index in ds.episode_indices(task_index=46)[:15]:
        r = ds.episode_rewards(task_index=46, episode_index=episode_index)
        six = np.nonzero(np.abs(r * 7 - 6) < 1e-3)[0]
        assert len(six) == 1, f"ep {episode_index}"
        assert not np.any(np.abs(r[six[0] + 1:]) > 1e-9), f"ep {episode_index}"

    lab = labels.episode_labels(ds.episode_rewards(46, 9200), D=7, s0_units=1)
    st = labels.progress_structure(lab.progress, D=7)
    assert st["max_step_frac"] == pytest.approx(6 / 7, abs=1e-3)


# --------------------------------------------------------------------------
# emitted schema -- pinned because downstream joins on it
# --------------------------------------------------------------------------

EXPECTED_SCHEMA = {
    "index": "int64",
    "episode_index": "int64",
    "frame_index": "int64",
    "task_index": "int32",
    "progress": "float",          # pyarrow's name for float32
    "satisfied_count": "float",
}


def test_emitted_label_schema_is_exactly_as_pinned(tmp_path, ds, rmap):
    """Column names, order and physical types. Downstream merges join on this.

    The init anchor changes the VALUES in `progress` and `satisfied_count`. It must not
    change the schema, the join keys, or the range.
    """
    labels.build_task_labels("cook_bacon", ds, rmap, tmp_path, max_episodes=5)
    schema = pq.read_schema(tmp_path / "cook_bacon" / "labels.parquet")

    assert schema.names == list(EXPECTED_SCHEMA)
    for name, want in EXPECTED_SCHEMA.items():
        assert str(schema.field(name).type) == want, name

    t = pq.read_table(tmp_path / "cook_bacon" / "labels.parquet").to_pandas()
    assert t.progress.min() >= 0.0 and t.progress.max() <= 1.0
    assert t.satisfied_count.max() <= 7.0
    np.testing.assert_allclose(t.satisfied_count, t.progress * 7, atol=1e-4)


def test_join_keys_are_unique_and_dense(tmp_path, ds, rmap):
    """(episode_index, frame_index) is the join key and must stay a primary key."""
    labels.build_task_labels("cook_bacon", ds, rmap, tmp_path, max_episodes=5)
    t = pq.read_table(tmp_path / "cook_bacon" / "labels.parquet").to_pandas()

    assert not t.duplicated(["episode_index", "frame_index"]).any()
    assert not t[["episode_index", "frame_index", "index", "progress"]].isna().any().any()
    for _, grp in t.groupby("episode_index"):
        assert grp.sort_values("frame_index").frame_index.tolist() == list(range(len(grp)))


def test_incomplete_demos_are_kept_by_default_and_droppable_on_request(tmp_path, ds, rmap):
    """An episode that ends short is correctly labelled, so it is kept unless asked."""
    kept = labels.build_task_labels("installing_a_fax_machine", ds, rmap, tmp_path,
                                    max_episodes=40)
    assert kept["episodes_ended_short"] > 0
    assert kept["demo_completion_rate"] < 1.0

    dropped = labels.build_task_labels("installing_a_fax_machine", ds, rmap, tmp_path,
                                       max_episodes=40, drop_incomplete=True)
    assert dropped["episodes_ended_short"] == 0
    assert dropped["demo_completion_rate"] == pytest.approx(1.0)
    assert dropped["drop_reasons"]["incomplete_demo"] == kept["episodes_ended_short"]
