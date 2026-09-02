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

def test_over_firing_episode_is_dropped_as_out_of_range(ds):
    """cook_bacon episode 9203 credits 8/7 units -> phi0 = -1/7, outside [0,1]."""
    r = ds.episode_rewards(task_index=46, episode_index=9203)
    ep = labels.episode_labels(r, D=7)
    assert not ep.valid
    assert ep.drop_reason == "phi0_out_of_range"


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
    assert "phi0" in msg and "14" in msg


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


def test_manifest_records_D_counts_drop_reasons_and_mean_phi0(tmp_path, ds, rmap):
    man = labels.build_task_labels("cook_bacon", ds, rmap, tmp_path, max_episodes=10)
    assert man["task"] == "cook_bacon"
    assert man["D"] == 7
    assert man["episodes_used"] + man["episodes_dropped"] == 10
    assert man["episodes_dropped"] >= 1
    assert "phi0_out_of_range" in man["drop_reasons"]
    assert man["mean_phi0"] == pytest.approx(0.0, abs=1e-6)


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
        return {int(r["ep"]): r for r in csv.DictReader(fh) if int(r["task"]) == task_index}


@pytest.mark.parametrize("task_index,task_name,D", [
    (0, "turning_on_radio", 1),
    (69, "vacuuming_floors", 1),
    (46, "cook_bacon", 7),
])
def test_pipeline_reproduces_the_committed_sweep(ds, task_index, task_name, D):
    """phi0, repair count and validity must match behavior1k_episode_stats.csv exactly.

    That CSV is the 5,677-episode measurement the whole recipe rests on. If this
    drifts, either the pipeline changed or the dataset was re-patched underneath us.
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
