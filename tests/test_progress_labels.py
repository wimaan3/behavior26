"""Prove the progress label reaches the model, on a dataset slice, without a GPU.

Three things have to hold or the contribution trains on nothing:

  1. The merged dataset still LOADS. LeRobot reads parquet through
     `datasets.Dataset.from_parquet(paths, features=<meta/info.json>)`, so an
     extra column that is not registered in info.json raises and makes the
     dataset unloadable for every config, baseline included.
  2. `progress` survives RepackTransform, which discards every key not named in
     the mapping.
  3. The join is right -- frame N gets frame N's label, not frame N+1's.

The fixture here is a real LeRobot-shaped root (data/chunk-000/file-000.parquet
plus meta/info.json) small enough to build in a temp dir. It does not exercise
video decoding or episode metadata, which the progress column does not touch.

    pytest tests/test_progress_labels.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "scripts" / "merge_progress_labels.py"

pytest.importorskip("pyarrow", reason="needs pyarrow to build the dataset fixture")
pytest.importorskip("pandas", reason="needs pandas for the join")

EPISODES = 2
FRAMES = 5


def _progress_of(episode: int, frame: int) -> float:
    """A label that is unique per (episode, frame), so a misaligned join shows up."""
    return round(0.1 * episode + 0.01 * frame, 4)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A minimal LeRobot root: low-dim parquet + meta/info.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "task_root"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()

    episode_index, frame_index, index = [], [], []
    for ep in range(EPISODES):
        for fr in range(FRAMES):
            episode_index.append(ep)
            frame_index.append(fr)
            index.append(ep * FRAMES + fr)

    table = pa.table(
        {
            "index": pa.array(index, pa.int64()),
            "episode_index": pa.array(episode_index, pa.int64()),
            "frame_index": pa.array(frame_index, pa.int64()),
            "task_index": pa.array([0] * len(index), pa.int64()),
            "timestamp": pa.array([i / 30.0 for i in index], pa.float32()),
            "action": pa.array([[float(i)] * 23 for i in index], pa.list_(pa.float32(), 23)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": EPISODES,
                "total_frames": EPISODES * FRAMES,
                "fps": 30,
                "features": {
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "action": {"dtype": "float32", "shape": [23], "names": None},
                },
            },
            indent=4,
        )
    )
    return root


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    import pandas as pd

    rows = [
        {"episode_index": ep, "frame_index": fr, "progress": _progress_of(ep, fr)}
        for ep in range(EPISODES)
        for fr in range(FRAMES)
    ]
    path = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _merge(dataset: Path, labels: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MERGE), "--dataset-root", str(dataset), "--labels", str(labels), *extra],
        capture_output=True,
        text=True,
    )


def _read_merged(out_root: Path):
    import pyarrow.parquet as pq

    paths = sorted((out_root / "data").glob("*/*.parquet"))
    assert paths, f"no parquet written under {out_root}/data"
    import pyarrow as pa

    return pa.concat_tables([pq.read_table(p) for p in paths]).to_pandas()


# ------------------------------------------------------------------- the join


def test_merge_joins_each_frame_to_its_own_label(dataset, labels, tmp_path):
    out = tmp_path / "merged"
    proc = _merge(dataset, labels, "--out-root", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    df = _read_merged(out)
    assert "progress" in df.columns
    assert len(df) == EPISODES * FRAMES

    # Every row, not just a spot check -- an off-by-one join would still put
    # plausible numbers in plausible places.
    for row in df.itertuples():
        expected = _progress_of(row.episode_index, row.frame_index)
        assert row.progress == pytest.approx(expected, abs=1e-6), (
            f"episode {row.episode_index} frame {row.frame_index}: "
            f"got {row.progress}, expected {expected}"
        )


def test_info_json_registers_the_column(dataset, labels, tmp_path):
    """Without this the dataset does not load at all -- see the reload test."""
    out = tmp_path / "merged"
    assert _merge(dataset, labels, "--out-root", str(out)).returncode == 0

    info = json.loads((out / "meta" / "info.json").read_text())
    assert "progress" in info["features"], info["features"].keys()
    # shape [1] is what makes LeRobot map it to a scalar rather than a length-1
    # sequence (get_hf_features_from_features).
    assert info["features"]["progress"]["shape"] == [1]
    assert info["features"]["progress"]["dtype"] == "float32"


def test_merged_dataset_still_loads_through_the_lerobot_path(dataset, labels, tmp_path):
    """The check that a wrong info.json would fail.

    This is LeRobot's actual load call: Dataset.from_parquet with the features
    built from meta/info.json.
    """
    hf = pytest.importorskip("datasets")

    out = tmp_path / "merged"
    assert _merge(dataset, labels, "--out-root", str(out)).returncode == 0

    info = json.loads((out / "meta" / "info.json").read_text())
    features = hf.Features(
        {
            k: (
                hf.Value(v["dtype"])
                if tuple(v["shape"]) == (1,)
                else hf.Sequence(length=v["shape"][0], feature=hf.Value(v["dtype"]))
            )
            for k, v in info["features"].items()
        }
    )
    paths = [str(p) for p in sorted((out / "data").glob("*/*.parquet"))]
    ds = hf.Dataset.from_parquet(paths, features=features)

    assert "progress" in ds.column_names
    row = ds[0]
    assert not isinstance(row["progress"], list), "progress must load as a scalar, not a length-1 list"
    assert row["progress"] == pytest.approx(_progress_of(0, 0), abs=1e-6)


def test_unregistered_column_would_break_the_whole_dataset(dataset, labels, tmp_path):
    """Pins WHY info.json must be patched: the failure is total, not partial.

    An extra parquet column with a stale schema does not get dropped; it raises,
    and the dataset stops loading for every config including the baseline.
    """
    hf = pytest.importorskip("datasets")

    out = tmp_path / "merged"
    assert _merge(dataset, labels, "--out-root", str(out)).returncode == 0

    stale = hf.Features(  # the pre-merge schema
        {
            "index": hf.Value("int64"),
            "episode_index": hf.Value("int64"),
            "frame_index": hf.Value("int64"),
            "task_index": hf.Value("int64"),
            "timestamp": hf.Value("float32"),
            "action": hf.Sequence(length=23, feature=hf.Value("float32")),
        }
    )
    paths = [str(p) for p in sorted((out / "data").glob("*/*.parquet"))]
    with pytest.raises(Exception) as exc:
        hf.Dataset.from_parquet(paths, features=stale)

    # datasets wraps the real cause, so its own message is generic
    # ("An error occurred while generating the dataset"). The CastError is in
    # the chain -- what matters is that it RAISES rather than silently dropping
    # the column, which would be far harder to notice.
    chain, err = [], exc.value
    while err is not None:
        chain.append(f"{type(err).__name__}: {err}")
        err = err.__cause__ or err.__context__
    joined = " | ".join(chain).lower()
    assert "cast" in joined or "column names" in joined, joined


# -------------------------------------------------------------- loud failures


def test_partial_labels_are_refused(dataset, labels, tmp_path):
    """A partial join is worse than no join: the head trains on filler."""
    import pandas as pd

    partial = tmp_path / "partial.parquet"
    df = pd.read_parquet(labels)
    df.drop(index=df.index[-3:]).to_parquet(partial)

    proc = _merge(dataset, partial, "--out-root", str(tmp_path / "m"))
    assert proc.returncode != 0
    assert "have no label" in (proc.stdout + proc.stderr)


def test_out_of_range_progress_is_refused(dataset, tmp_path):
    """BCE is undefined outside [0, 1]."""
    import pandas as pd

    bad = tmp_path / "bad.parquet"
    pd.DataFrame(
        [
            {"episode_index": ep, "frame_index": fr, "progress": 1.5 if (ep, fr) == (0, 0) else 0.5}
            for ep in range(EPISODES)
            for fr in range(FRAMES)
        ]
    ).to_parquet(bad)

    proc = _merge(dataset, bad, "--out-root", str(tmp_path / "m"))
    assert proc.returncode != 0
    assert "[0, 1]" in (proc.stdout + proc.stderr)


def test_duplicate_label_keys_are_refused(dataset, labels, tmp_path):
    import pandas as pd

    dup = tmp_path / "dup.parquet"
    df = pd.read_parquet(labels)
    pd.concat([df, df.head(1)]).to_parquet(dup)

    proc = _merge(dataset, dup, "--out-root", str(tmp_path / "m"))
    assert proc.returncode != 0
    assert "duplicate" in (proc.stdout + proc.stderr).lower()


def test_merging_twice_is_refused(dataset, labels, tmp_path):
    out = tmp_path / "merged"
    assert _merge(dataset, labels, "--out-root", str(out)).returncode == 0
    # Re-merging the already-merged root would double-add the column.
    proc = _merge(out, labels, "--out-root", str(tmp_path / "again"))
    assert proc.returncode != 0
    assert "already has a 'progress' column" in (proc.stdout + proc.stderr)


def test_original_dataset_is_untouched(dataset, labels, tmp_path):
    """The baseline arm must provably read the original bytes."""
    before = {
        p: p.read_bytes() for p in sorted((dataset / "data").rglob("*.parquet"))
    }
    info_before = (dataset / "meta" / "info.json").read_text()

    assert _merge(dataset, labels, "--out-root", str(tmp_path / "merged")).returncode == 0

    for path, blob in before.items():
        assert path.read_bytes() == blob, f"{path} was modified"
    assert (dataset / "meta" / "info.json").read_text() == info_before


# ----------------------------------------------------- the repack forwarding


def test_repack_transform_forwards_progress():
    """openpi's real RepackTransform, with the mapping progress_key builds.

    RepackTransform discards every key not in the mapping, so this is the step
    that decides whether the column reaches the model at all.
    """
    from tests.openpi_fixture import require_openpi

    require_openpi()  # skips cleanly if the fork or jax is missing
    from openpi import transforms as _transforms  # noqa: PLC0415

    mapping = {
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
        "progress": "progress",  # what LeRobotB1KDataConfig.progress_key adds
    }
    item = {
        "observation.state": [0.0] * 57,
        "action": [0.0] * 23,
        "prompt": "do the thing",
        "progress": 0.42,
        "episode_index": 0,  # extra columns must be dropped, not crash
        "frame_index": 3,
    }
    out = _transforms.RepackTransform(mapping)(item)

    assert out["progress"] == pytest.approx(0.42)
    assert "episode_index" not in out, "repack should drop unmapped columns"


def test_repack_without_progress_key_drops_the_column():
    """The default config path: progress_key=None means no mapping entry.

    Confirms the label is dropped rather than leaking into the baseline arm.
    """
    from tests.openpi_fixture import require_openpi

    require_openpi()
    from openpi import transforms as _transforms  # noqa: PLC0415

    mapping = {"observation/state": "observation.state", "actions": "action"}
    out = _transforms.RepackTransform(mapping)(
        {"observation.state": [0.0] * 57, "action": [0.0] * 23, "progress": 0.42}
    )
    assert "progress" not in out
