"""Tests for analysis.reward.fetch_rewards, run against REAL LeRobot parquet.

The mirror exists so `D` and `phi0` can be measured on all 20,000 episodes without
pulling ~70 GB of observations. It is only trustworthy if a mirrored shard is
indistinguishable from the real one on the columns it keeps -- and in particular if
the synthesised `index` equals the real `index`, because `index` is the join key the
training loader uses. A wrong join key is worse than no labels: it silently pairs
frames with other frames' progress.

Requires the same local pull as test_labels.py. Skips if absent.
"""
import os
import pathlib
import shutil

import pandas as pd
import pyarrow.parquet as pq
import pytest

from analysis.reward import fetch_rewards as fr
from analysis.reward import labels

DATA_ROOT = pathlib.Path(os.environ.get("BEHAVIOR_DATA_ROOT", pathlib.Path.home() / "behavior-data"))

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "meta" / "info.json").exists(),
    reason=f"no local LeRobot pull at {DATA_ROOT}",
)

# Shards held locally in full, one per interesting task (see test_labels.py).
LOCAL_SHARDS = [(0, 0), (11, 0), (46, 0)]


@pytest.fixture(scope="module")
def meta():
    return fr.episode_map(DATA_ROOT)


@pytest.fixture(scope="module")
def mirror(tmp_path_factory, meta):
    out = tmp_path_factory.mktemp("mirror")
    # build_shard writes data/ only; meta/ is what makes the mirror a usable Dataset.
    shutil.copytree(DATA_ROOT / "meta", out / "meta", dirs_exist_ok=True)
    for chunk, fidx in LOCAL_SHARDS:
        if (DATA_ROOT / "data" / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.parquet").exists():
            fr.build_shard(chunk, fidx, DATA_ROOT, out, meta)
    return out


def _pair(chunk, fidx, mirror):
    rel = f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet"
    if not (DATA_ROOT / rel).exists():
        pytest.skip(f"{rel} not held locally")
    return DATA_ROOT / rel, mirror / rel


@pytest.mark.parametrize("chunk,fidx", LOCAL_SHARDS)
def test_synthesised_index_equals_the_real_index(chunk, fidx, mirror):
    """The whole mirror rests on this: index == dataset_from_index + frame_index."""
    real_p, mir_p = _pair(chunk, fidx, mirror)
    real = pq.read_table(real_p, columns=["index"]).to_pandas()
    mir = pq.read_table(mir_p, columns=["index"]).to_pandas()
    assert real.equals(mir)


@pytest.mark.parametrize("chunk,fidx", LOCAL_SHARDS)
def test_kept_columns_are_bit_identical(chunk, fidx, mirror):
    real_p, mir_p = _pair(chunk, fidx, mirror)
    cols = ["index", "episode_index", "frame_index",
            "next.reward", "next.terminated", "next.truncated"]
    real = pq.read_table(real_p, columns=cols).to_pandas()
    mir = pq.read_table(mir_p, columns=cols).to_pandas()
    pd.testing.assert_frame_equal(real, mir)


def test_mirror_measures_the_same_D_and_phi0_as_the_real_pull(mirror, meta):
    """A Dataset over the mirror must agree with one over the real pull."""
    real_ds = labels.Dataset(DATA_ROOT)
    mir_ds = labels.Dataset(mirror)
    for chunk, _ in LOCAL_SHARDS:
        eps = meta[(meta["data/chunk_index"] == chunk)
                   & (meta["data/file_index"] == 0)].episode_index.tolist()
        a = real_ds.read_columns(chunk, labels.REWARD_COLUMNS, eps)
        b = mir_ds.read_columns(chunk, labels.REWARD_COLUMNS, eps)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_build_shard_rejects_a_row_count_that_disagrees_with_meta(tmp_path, meta):
    """Integrity gate: if meta and the shard disagree, the synthesised index would be
    wrong, so the build must fail loudly rather than write a bad join key."""
    bad = meta.copy()
    bad.loc[bad.index[0], "length"] = int(bad.loc[bad.index[0], "length"]) + 1
    with pytest.raises(ValueError, match="rows, meta says"):
        fr.build_shard(int(bad.iloc[0]["data/chunk_index"]),
                       int(bad.iloc[0]["data/file_index"]), DATA_ROOT, tmp_path, bad)


def test_episode_map_covers_the_whole_corpus(meta):
    assert len(meta) == 20_000
    assert meta.task_index.nunique() == 100
    assert (meta.groupby("task_index").size() == 200).all()
