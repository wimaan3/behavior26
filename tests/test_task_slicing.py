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
EPISODES_PER_TASK = 3   # >=3 so a dropped MIDDLE episode leaves a gap
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

    # Real v3 episode metadata carries GLOBAL frame ranges and video offsets. The
    # ranges are what the reader clamps delta-timestamp queries into, so they must
    # be in the fixture or the third index trap is invisible to these tests.
    eps = pd.DataFrame([
        {"episode_index": g, "task_index": t, "length": FRAMES,
         "dataset_from_index": g * FRAMES, "dataset_to_index": g * FRAMES + FRAMES,
         "videos/observation.images.head/chunk_index": 0,
         "videos/observation.images.head/file_index": 0,
         "videos/observation.images.head/from_timestamp": g * FRAMES / 30.0}
        for g, _, t in layout
    ])
    vdir = root / "videos" / "observation.images.head" / "chunk-000"
    vdir.mkdir(parents=True)
    (vdir / "file-000.mp4").write_bytes(b"not-really-an-mp4")
    pq.write_table(pa.Table.from_pandas(eps, preserve_index=False),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    tasks = pd.DataFrame({"task_index": list(TASKS.values())},
                         index=pd.Index(list(TASKS.keys()), name="task"))
    pq.write_table(pa.Table.from_pandas(tasks), root / "meta" / "tasks.parquet")

    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 30,
        "total_episodes": ep, "total_frames": len(rows), "total_tasks": len(TASKS),
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
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


# ------------------------------------------------ episode frame ranges (trap 3)

def _read_episodes(dest: Path):
    import pandas as pd
    import pyarrow.parquet as pq
    return pd.concat([pq.read_table(p).to_pandas()
                      for p in sorted((dest / "meta" / "episodes").glob("*/*.parquet"))],
                     ignore_index=True)


def test_episode_frame_ranges_match_the_renumbered_index(source, tmp_path):
    """The reader clamps every delta-timestamp query into
    [dataset_from_index, dataset_to_index) and uses the result as a ROW POSITION.
    Left global after `index` is renumbered, action chunks come from the wrong
    frames -- loudly for most tasks, SILENTLY for any task whose global offset
    falls inside the slice. Every episode's range must cover exactly its rows."""
    dest = tmp_path / "out" / "beta_task"
    slice_dataset(source, "beta_task", tmp_path / "out")
    data, eps = _read_slice(dest), _read_episodes(dest)
    assert sorted(data["index"]) == list(range(len(data))), "index must equal row position"
    for ep in eps.itertuples():
        rows = data[data["episode_index"] == ep.episode_index]
        assert sorted(rows["index"]) == list(range(ep.dataset_from_index, ep.dataset_to_index)), (
            f"episode {ep.episode_index}: range [{ep.dataset_from_index},{ep.dataset_to_index}) "
            f"does not match its rows {sorted(rows['index'])}"
        )


def test_referenced_videos_exist_at_their_original_paths(source, tmp_path):
    dest = tmp_path / "out" / "beta_task"
    m = slice_dataset(source, "beta_task", tmp_path / "out")
    assert (dest / "videos" / "observation.images.head" / "chunk-000" / "file-000.mp4").exists()
    assert m["video_files"] == 1


def test_a_missing_referenced_video_is_refused(source, tmp_path):
    """Otherwise the slice loads and dies on the first sample -- on a GPU box."""
    (source / "videos" / "observation.images.head" / "chunk-000" / "file-000.mp4").unlink()
    with pytest.raises(SystemExit, match="referenced video"):
        slice_dataset(source, "beta_task", tmp_path / "out")


# ------------------------------------------------- THE ARM-B POISONING TEST

def test_progress_labels_land_on_the_globally_correct_episodes(source, tmp_path):
    """The test that matters most in this suite.

    The Jetson labels against the PRISTINE corpus, so its sidecars are keyed on the
    global episode index. After slicing, `episode_index` means something else. Joining on
    the raw column would attach each label to whichever local episode happens to
    share the number -- arm A unaffected, arm B trained against labels belonging
    to DIFFERENT demonstrations, and dQ meaningless with nothing raising.

    Joining on `global_episode_index` must put every label back on its own frames.
    """
    import pandas as pd

    dest = tmp_path / "out" / "gamma_task"
    slice_dataset(source, "gamma_task", tmp_path / "out")

    # Label EVERY global episode -- the WORST case, deliberately. The real Jetson
    # sidecars are per task (origin/jetson/labels: labels/<task>/labels.parquet,
    # global episodes task_index*200+i), so for the current pair a wrong join on
    # local 0..199 looks up turning_on_radio's range, finds nothing, and fails
    # loudly. It turns SILENT once arm B concatenates sidecars that cover global
    # 0..N-1 -- plausible as k grows. A fixture covering only this task would let
    # a wrong join fail on missing labels and pass this test for the wrong reason.
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


def test_drop_unlabelled_works_when_joining_on_the_global_column(source, tmp_path):
    """Protocol mode (AB_PROTOCOL 3.6) is --drop-unlabelled, and arm B joins on
    global_episode_index. The merge selected ONLY the join columns out of each
    parquet, so `episode_index` -- which dropping needs, because episodes are the
    unit -- was not there, and the real run died with
    "--drop-unlabelled needs an episode_index column". Found on real data
    2026-09-14 after the synthetic test passed without the flag.
    """
    import pandas as pd

    dest = tmp_path / "out" / "gamma_task"
    slice_dataset(source, "gamma_task", tmp_path / "out")
    gamma = sorted(g for g, n, _ in _layout_of(source) if n == "gamma_task")
    # Label everything EXCEPT the last frame of gamma's second episode.
    rows = [
        {"episode_index": g, "frame_index": fr, "progress": _progress_of(g, fr)}
        for g, _, _ in _layout_of(source) for fr in range(FRAMES)
        if not (g == gamma[1] and fr == FRAMES - 1)
    ]
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(labels_path, index=False)

    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "merge_progress_labels.py"),
         "--dataset-root", str(dest), "--labels", str(labels_path),
         "--out-root", str(tmp_path / "merged"),
         "--join-on", GLOBAL_EP_COL, "frame_index",
         "--labels-join-on", "episode_index", "frame_index",
         "--drop-unlabelled"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    out = tmp_path / "merged" / dest.name if (tmp_path / "merged" / dest.name).exists() else tmp_path / "merged"
    merged = _read_slice(out)
    # The unlabelled episode is gone WHOLE, and the other survives intact.
    assert gamma[1] not in set(merged[GLOBAL_EP_COL].unique()), "wrong episode dropped"
    assert len(merged) == FRAMES * (EPISODES_PER_TASK - 1)


# ------------------------------------------- compaction after --drop-unlabelled

def _merge(dest: Path, labels_path: Path, out: Path, *, drop=True):
    cmd = [sys.executable, str(REPO / "scripts" / "merge_progress_labels.py"),
           "--dataset-root", str(dest), "--labels", str(labels_path), "--out-root", str(out),
           "--join-on", GLOBAL_EP_COL, "frame_index",
           "--labels-join-on", "episode_index", "frame_index"]
    if drop:
        cmd.append("--drop-unlabelled")
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return out if (out / "data").exists() else out / dest.name


@pytest.fixture
def sliced_with_a_hole(source, tmp_path):
    """A gamma slice plus labels missing one frame of its MIDDLE episode."""
    import pandas as pd
    dest = tmp_path / "out" / "gamma_task"
    slice_dataset(source, "gamma_task", tmp_path / "out")
    gamma = sorted(g for g, n, _ in _layout_of(source) if n == "gamma_task")
    hole = gamma[1]
    rows = [{"episode_index": g, "frame_index": fr, "progress": _progress_of(g, fr)}
            for g, _, _ in _layout_of(source) for fr in range(FRAMES)
            if not (g == hole and fr == FRAMES - 1)]
    lp = tmp_path / "labels.parquet"
    pd.DataFrame(rows).to_parquet(lp, index=False)
    return dest, lp, gamma, hole


def test_dropping_an_episode_leaves_a_dense_root(sliced_with_a_hole, tmp_path):
    """LeRobot v3 indexes episodes POSITIONALLY. Dropping the middle episode and
    leaving the numbering alone -- correct for v2 -- puts every later episode off
    by one, breaks the frame ranges, and sends LeRobot to the Hub for a repo that
    does not exist. Measured on the real coffee-station root 2026-09-14."""
    dest, lp, gamma, hole = sliced_with_a_hole
    out = _merge(dest, lp, tmp_path / "merged")
    data, eps = _read_slice(out), _read_episodes(out)
    assert sorted(data["episode_index"].unique()) == [0, 1], "episode_index must be dense after a drop"
    assert sorted(data["index"]) == list(range(len(data))), "index must be dense after a drop"
    assert len(eps) == 2, "meta/episodes must be pruned, not left listing the dropped episode"
    assert sorted(eps["episode_index"]) == [0, 1]
    for ep in eps.itertuples():
        rows = data[data["episode_index"] == ep.episode_index]
        assert sorted(rows["index"]) == list(range(ep.dataset_from_index, ep.dataset_to_index))


def test_compaction_preserves_the_global_mapping(sliced_with_a_hole, tmp_path):
    """The whole point of the column: after TWO renumberings (slice, then
    compaction) a label must still be traceable to its original demonstration."""
    import numpy as np, pandas as pd
    dest, lp, gamma, hole = sliced_with_a_hole
    out = _merge(dest, lp, tmp_path / "merged")
    data = _read_slice(out)
    assert sorted(data[GLOBAL_EP_COL].unique()) == [gamma[0], gamma[2]], "the dropped episode is the hole"
    side = pd.read_parquet(lp).set_index(["episode_index", "frame_index"])["progress"]
    want = side.reindex(pd.MultiIndex.from_arrays([data[GLOBAL_EP_COL], data["frame_index"]])).to_numpy()
    assert np.allclose(data["progress"].to_numpy(dtype="float32"), want.astype("float32"), atol=1e-6)


def test_info_totals_match_the_compacted_rows(sliced_with_a_hole, tmp_path):
    dest, lp, gamma, hole = sliced_with_a_hole
    out = _merge(dest, lp, tmp_path / "merged")
    info = json.loads((out / "meta" / "info.json").read_text())
    data = _read_slice(out)
    assert info["total_episodes"] == data["episode_index"].nunique() == 2
    assert info["total_frames"] == len(data)


def test_the_filter_manifest_records_what_a_reader_must_check(sliced_with_a_hole, tmp_path):
    """meta/progress_filter.json is the artifact the arm-parity guard reads, so it
    has to carry the post-compaction episode count, not just what was dropped."""
    dest, lp, gamma, hole = sliced_with_a_hole
    out = _merge(dest, lp, tmp_path / "merged")
    m = json.loads((out / "meta" / "progress_filter.json").read_text())
    assert m["episodes_out"] == 2
    assert m["compacted"] is True
    # local (pre-compaction) -> new, and pristine global -> new. Both, because
    # "episode_remap" alone is ambiguous and an auditor reading the wrong one
    # checks nothing.
    assert m["episode_remap_local"] == {"0": 0, "2": 1}
    assert m["global_episodes_kept"] == {str(gamma[0]): 0, str(gamma[2]): 1}
