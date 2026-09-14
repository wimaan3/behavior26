"""Slicing the combined corpus into per-task datasets.

Two index rules, and they differ for reasons the tests state:
  * task_index  is preserved globally  (prompts resolve by integer lookup)
  * episode_index is renumbered densely (LeRobot v3 indexes episodes positionally)

The second is what would silently poison arm B, so the last test in this file is
the one that matters most in the suite: it asserts progress labels land on the
GLOBALLY correct demonstrations after a slice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.slice_task_dataset import GLOBAL_EP_COL, slice_dataset  # noqa: E402

# A source with THREE tasks whose episode indices interleave, so a slice cannot
# accidentally look correct by being a contiguous prefix.
TASKS = {"alpha_task": 3, "beta_task": 7, "gamma_task": 11}
EPISODES_PER_TASK = 2
FRAMES = 4


def _progress_of(global_ep: int, frame: int) -> float:
    """Unique per (global episode, frame), so a misaligned join is visible."""
    return round(0.001 * global_ep + 0.01 * frame, 5)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "combined"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    rows, ep = [], 0
    layout = []                      # (global_ep, task_name, task_index)
    for i in range(EPISODES_PER_TASK):
        for name, t_idx in TASKS.items():          # interleaved, not grouped
            layout.append((ep, name, t_idx))
            for fr in range(FRAMES):
                rows.append({
                    "index": len(rows),
                    "episode_index": ep,
                    "frame_index": fr,
                    "task_index": t_idx,
                    "observation.state": float(ep * 100 + fr),
                })
            ep += 1
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   root / "data" / "chunk-000" / "file-000.parquet")

    eps = pd.DataFrame([
        {"episode_index": g, "task_index": t, "length": FRAMES,
         "videos/observation.images.head/chunk_index": 0,
         "videos/observation.images.head/file_index": 0}
        for g, _, t in layout
    ])
    pq.write_table(pa.Table.from_pandas(eps, preserve_index=False),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    tasks = pd.DataFrame({"task_index": list(TASKS.values())},
                         index=pd.Index(list(TASKS.keys()), name="task"))
    pq.write_table(pa.Table.from_pandas(tasks), root / "meta" / "tasks.parquet")

    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30,
        "total_episodes": ep, "total_frames": len(rows), "total_tasks": len(TASKS),
        "features": {"observation.state": {"dtype": "float32", "shape": [1], "names": None}},
    }))
    return root


def _read_slice(dest: Path):
    import pandas as pd
    import pyarrow.parquet as pq
    frames = [pq.read_table(p).to_pandas() for p in sorted((dest / "data").glob("*/*.parquet"))]
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- task_index

def test_task_index_is_preserved_globally(source, tmp_path):
    """Renumbering to 0 would make every slice emit 0, and every sample in a
    k-task run would get the FIRST task's prompt -- silently."""
    slice_dataset(source, "beta_task", tmp_path / "out")
    df = _read_slice(tmp_path / "out" / "beta_task")
    assert set(df["task_index"].unique()) == {TASKS["beta_task"]} == {7}


def test_the_full_task_table_ships_with_every_slice(source, tmp_path):
    """openpi builds the prompt map from repo_id[0]'s metadata alone, so slice 0's
    table must be able to resolve every other slice's task_index."""
    import pyarrow.parquet as pq
    slice_dataset(source, "alpha_task", tmp_path / "out")
    tasks = pq.read_table(tmp_path / "out" / "alpha_task" / "meta" / "tasks.parquet").to_pandas()
    assert set(tasks["task_index"]) == set(TASKS.values()), "all 3 tasks must remain resolvable"


# -------------------------------------------------------------- episode_index

def test_episode_index_is_dense_and_zero_based(source, tmp_path):
    """LeRobot v3 indexes episodes POSITIONALLY: `self.episodes[ep_index]` bounded
    by len(). A slice carrying global indices raises IndexError on video lookup."""
    slice_dataset(source, "gamma_task", tmp_path / "out")
    df = _read_slice(tmp_path / "out" / "gamma_task")
    assert sorted(df["episode_index"].unique()) == list(range(EPISODES_PER_TASK))


def test_the_global_episode_index_survives_as_a_column(source, tmp_path):
    """The mapping rides WITH the rows. A sidecar file can be paired with the
    wrong slice, lost in a copy, or go stale after a re-slice; a column cannot."""
    slice_dataset(source, "gamma_task", tmp_path / "out")
    df = _read_slice(tmp_path / "out" / "gamma_task")
    expected = sorted(g for g, n, _ in
                      [(e, n, t) for e, n, t in _layout_of(source)] if n == "gamma_task")
    assert sorted(df[GLOBAL_EP_COL].unique()) == expected
    # and the two indices must genuinely differ, or the test proves nothing
    assert (df[GLOBAL_EP_COL] != df["episode_index"]).any()


def _layout_of(source: Path):
    import pyarrow.parquet as pq
    eps = pq.read_table(source / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    inv = {v: k for k, v in TASKS.items()}
    return [(int(r.episode_index), inv[int(r.task_index)], int(r.task_index)) for r in eps.itertuples()]


def test_global_column_is_registered_in_info_features(source, tmp_path):
    """An unregistered column makes the dataset unloadable -- the schema is built
    from this dict. Same trap `progress` has."""
    slice_dataset(source, "beta_task", tmp_path / "out")
    info = json.loads((tmp_path / "out" / "beta_task" / "meta" / "info.json").read_text())
    assert GLOBAL_EP_COL in info["features"]
    assert info["features"][GLOBAL_EP_COL]["dtype"] == "int64"


def test_chunk_numbering_is_preserved(source, tmp_path):
    """chunk/file indices are read from each episode row, not computed, so the
    originals are kept -- provenance stays obvious and videos resolve unchanged."""
    slice_dataset(source, "beta_task", tmp_path / "out")
    got = sorted(p.relative_to(tmp_path / "out" / "beta_task").as_posix()
                 for p in (tmp_path / "out" / "beta_task" / "data").glob("*/*.parquet"))
    assert got == ["data/chunk-000/file-000.parquet"]


def test_slicing_a_slice_is_refused(source, tmp_path):
    """The second slice would renumber an already-renumbered episode_index and
    silently destroy the global mapping."""
    slice_dataset(source, "beta_task", tmp_path / "out")
    with pytest.raises(SystemExit, match="refusing to slice a slice"):
        slice_dataset(tmp_path / "out" / "beta_task", "beta_task", tmp_path / "out2")


def test_frame_counter_stays_dense(source, tmp_path):
    slice_dataset(source, "alpha_task", tmp_path / "out")
    df = _read_slice(tmp_path / "out" / "alpha_task")
    assert sorted(df["index"].tolist()) == list(range(len(df)))


# ------------------------------------------------- THE ARM-B POISONING TEST

def test_progress_labels_land_on_the_globally_correct_episodes(source, tmp_path):
    """The test that matters most in this suite.

    The Jetson labels the PRISTINE corpus, so its sidecar is keyed on the global
    episode index. After slicing, `episode_index` means something else. Joining on
    the raw column would attach each label to whichever local episode happens to
    share the number -- arm A unaffected, arm B trained against labels belonging
    to DIFFERENT demonstrations, and dQ meaningless with nothing raising.

    Joining on `global_episode_index` must put every label back on its own frames.
    """
    import pandas as pd

    dest = tmp_path / "out" / "gamma_task"
    slice_dataset(source, "gamma_task", tmp_path / "out")

    # Label EVERY global episode, as the Jetson does -- it labels the pristine
    # corpus, not one task. This is what makes a wrong join SILENT: for any local
    # index i there is always a label for global episode i, belonging to whichever
    # task owns it. If the sidecar only covered this task, a wrong join would
    # error on missing labels and the test would pass for the wrong reason.
    all_eps = sorted(g for g, _, _ in _layout_of(source))
    labels = pd.DataFrame([
        {"episode_index": g, "frame_index": fr, "progress": _progress_of(g, fr)}
        for g in all_eps for fr in range(FRAMES)
    ])
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)

    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "merge_progress_labels.py"),
         "--dataset-root", str(dest), "--labels", str(labels_path),
         "--out-root", str(tmp_path / "merged"),
         "--join-on", GLOBAL_EP_COL, "frame_index",
         "--labels-join-on", "episode_index", "frame_index"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    merged = _read_slice(tmp_path / "merged" / dest.name) if (tmp_path / "merged" / dest.name).exists() \
        else _read_slice(tmp_path / "merged")
    assert "progress" in merged.columns
    for row in merged.itertuples():
        want = _progress_of(int(getattr(row, GLOBAL_EP_COL)), int(row.frame_index))
        assert abs(float(row.progress) - want) < 1e-6, (
            f"label misaligned: local ep {row.episode_index}, global "
            f"{getattr(row, GLOBAL_EP_COL)}, frame {row.frame_index} -- "
            f"got {row.progress}, want {want}. This is arm-B poisoning."
        )
