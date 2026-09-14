#!/usr/bin/env python3
"""Slice one task out of the combined BEHAVIOR-1K corpus into a standalone
LeRobot v3.0 dataset that openpi's `pi05_b1k` configs can load.

WHY THIS EXISTS
---------------
`behavior-1k/2026-challenge-demos` is ONE 100-task LeRobot dataset. openpi's b1k
configs want a dataset per task -- `repo_id` names it and the loader resolves
`<dataset_root>/<repo_id>`. No per-task datasets are published (checked against
the `behavior-1k` org), so this is step 1 of the fork's docs/b1k.md and it is on
us to write. See docs/DATASET_SLICING.md.

THE TWO INDEX RULES, AND WHY THEY DIFFER
----------------------------------------
`task_index` is PRESERVED GLOBALLY and `meta/tasks.parquet` is copied whole.
    `PromptFromLeRobotTask` maps prompts by integer lookup, and when `repo_id` is
    a list openpi builds that map from `repo_id[0]`'s metadata ALONE. Renumbering
    each slice to 0 would make every slice emit 0, so every sample in a k-task run
    would receive the FIRST task's prompt -- silently, with no error, training on
    systematically mislabelled instructions. Keeping global indices plus the full
    table makes index->name identical across slices and turns a genuine mismatch
    into a loud KeyError.

`episode_index` is RENUMBERED to a dense 0..N-1, because LeRobot v3 requires it:

        if ep_index >= len(self.episodes):
            raise IndexError(...)
        ep = self.episodes[ep_index]     # positional row lookup

    `load_episodes` returns a `datasets.Dataset`, and the reader's cache check
    asks for `set(range(total_episodes))`. Dense zero-based indexing is
    structural, not a convention -- a slice carrying global indices 4000..4049
    fails on both counts.

    That renumbering is exactly what would silently re-point the Jetson's
    (episode_index, frame_index) progress sidecars at the WRONG demonstrations:
    arm A unaffected, arm B trained against mismatched labels, dQ meaningless and
    nothing raising. So the original index is carried as a COLUMN,
    `global_episode_index`, registered in `meta/info.json` features. A column
    cannot be separated from the rows it describes the way a sidecar file can --
    it cannot be paired with the wrong slice, lost in a copy, or go stale after a
    re-slice. merge_progress_labels.py then joins on
    (global_episode_index, frame_index) and fails loudly if the column is absent.

`chunk_index` / `file_index` are LEFT ALONE. They are read from each episode's
metadata row rather than computed from the episode index, and data files load by
a tree walk, so original chunk numbering is preserved -- which also keeps
provenance obvious and lets the video files be copied at unchanged paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

GLOBAL_EP_COL = "global_episode_index"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_task_index(source: Path, task: str) -> int:
    """Global task_index for a task NAME, read from meta/tasks.parquet."""
    import pandas as pd

    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise SystemExit(f"no {tasks_path}; is {source} a LeRobot v3 root?")
    df = pd.read_parquet(tasks_path)
    names = [str(i) for i in df.index]
    if task not in names:
        near = [n for n in names if task.lower() in n.lower()][:5]
        raise SystemExit(
            f"task {task!r} not in {tasks_path}. {len(names)} tasks available."
            + (f" Did you mean: {near}?" if near else "")
        )
    return int(df.loc[task, "task_index"])


def slice_dataset(source: Path, task: str, out_root: Path, *, overwrite: bool = False) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    task_index = resolve_task_index(source, task)
    dest = out_root / task
    if dest.exists():
        if not overwrite:
            raise SystemExit(f"{dest} exists; pass --overwrite to replace it")
        shutil.rmtree(dest)

    data_files = sorted((source / "data").glob("*/*.parquet"))
    if not data_files:
        raise SystemExit(f"no data/*/*.parquet under {source}")

    # PASS 1 -- find this task's episodes, in a STABLE order. Sorted by the
    # global index so the positional renumbering is deterministic and a re-slice
    # of the same source reproduces the same mapping.
    global_eps: set[int] = set()
    for path in data_files:
        tbl = pq.read_table(path, columns=["task_index", "episode_index"])
        df = tbl.to_pandas()
        global_eps.update(int(e) for e in df.loc[df["task_index"] == task_index, "episode_index"].unique())
    if not global_eps:
        raise SystemExit(f"no rows with task_index={task_index} ({task!r}) in {source}/data")
    ordered = sorted(global_eps)
    remap = {g: i for i, g in enumerate(ordered)}

    # PASS 2 -- rewrite the rows that belong to this task.
    (dest / "meta").mkdir(parents=True)
    rows_out = 0
    frame_cursor = 0
    ranges: dict[int, tuple[int, int]] = {}     # new episode_index -> dense [from, to)
    written: list[Path] = []
    for path in data_files:
        tbl = pq.read_table(path)
        df = tbl.to_pandas()
        keep = df["task_index"] == task_index
        if not keep.any():
            continue
        sub = df.loc[keep].copy()
        if GLOBAL_EP_COL in sub.columns:
            raise SystemExit(
                f"{path} already has a {GLOBAL_EP_COL!r} column -- refusing to slice a slice. "
                "Point --source at the pristine combined corpus."
            )
        sub = sub.sort_values(["episode_index", "frame_index"], kind="stable")
        sub[GLOBAL_EP_COL] = sub["episode_index"].astype("int64")
        sub["episode_index"] = sub["episode_index"].map(remap).astype("int64")
        if "index" not in sub.columns:
            raise SystemExit(f"{path} has no 'index' column; not a LeRobot v3 data file")
        # `index` is the dataset-wide frame counter and must stay dense: the v3
        # reader uses it AS A ROW POSITION (_absolute_to_relative_idx is None when
        # every episode is loaded).
        sub["index"] = range(frame_cursor, frame_cursor + len(sub))
        frame_cursor += len(sub)
        for ep_local, grp in sub.groupby("episode_index", sort=False):
            lo, hi = int(grp["index"].min()), int(grp["index"].max()) + 1
            if ep_local in ranges:
                raise SystemExit(f"episode {ep_local} spans more than one data file; unsupported")
            ranges[int(ep_local)] = (lo, hi)
        rel = path.relative_to(source)          # data/chunk-NNN/file-NNN.parquet
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), target)
        written.append(target)
        rows_out += len(sub)

    # meta/tasks.parquet -- copied WHOLE and UNCHANGED. See the module docstring.
    shutil.copy2(source / "meta" / "tasks.parquet", dest / "meta" / "tasks.parquet")

    # meta/episodes -- same renumbering, same preservation of chunk/file indices.
    video_refs: set[tuple[str, int, int]] = set()
    src_eps_dir = source / "meta" / "episodes"
    if src_eps_dir.is_dir():
        for path in sorted(src_eps_dir.glob("*/*.parquet")):
            edf = pq.read_table(path).to_pandas()
            if "episode_index" not in edf.columns:
                continue
            keep = edf["episode_index"].isin(ordered)
            if not keep.any():
                continue
            sub = edf.loc[keep].copy().sort_values("episode_index", kind="stable")
            sub[GLOBAL_EP_COL] = sub["episode_index"].astype("int64")
            sub["episode_index"] = sub["episode_index"].map(remap).astype("int64")
            # THE THIRD INDEX TRAP. dataset_from_index / dataset_to_index are in the
            # same space as `index`, and the reader clamps every delta-timestamp query
            # into [from, to) and then uses the result as a ROW POSITION:
            #     query = max(ep_start, min(ep_end - 1, abs_idx + delta))
            # Left at their global values after `index` is renumbered, every action
            # chunk would be drawn from the wrong frames -- an IndexError for most
            # tasks, but SILENTLY wrong action targets for any task whose global
            # offset happens to fall inside the slice's length.
            if {"dataset_from_index", "dataset_to_index"} <= set(sub.columns):
                missing = sorted(set(sub["episode_index"]) - set(ranges))
                if missing:
                    raise SystemExit(f"episode metadata for {missing} but no data rows; source is inconsistent")
                sub["dataset_from_index"] = sub["episode_index"].map(lambda e: ranges[e][0]).astype("int64")
                sub["dataset_to_index"] = sub["episode_index"].map(lambda e: ranges[e][1]).astype("int64")
                for col in ("length",):
                    if col in sub.columns:
                        bad = sub[(sub["dataset_to_index"] - sub["dataset_from_index"]) != sub[col]]
                        if len(bad):
                            raise SystemExit(
                                f"episode length disagrees with its data rows for episodes "
                                f"{bad['episode_index'].tolist()[:5]}; refusing to write a slice that lies"
                            )
            for vcol in [c for c in sub.columns if c.startswith("videos/") and c.endswith("/chunk_index")]:
                key = vcol[len("videos/"):-len("/chunk_index")]
                for ci, fi in set(zip(sub[vcol], sub[f"videos/{key}/file_index"])):
                    video_refs.add((key, int(ci), int(fi)))
            rel = path.relative_to(source)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), target)
            written.append(target)

    # videos -- every file an episode references, at its ORIGINAL chunk/file path.
    # Episodes locate their frames inside a shared mp4 by from_timestamp, which is
    # untouched, so whole files are copied rather than cut. Hard-linked when source
    # and destination share a filesystem, so a slice costs no extra disk.
    info_src = json.loads((source / "meta" / "info.json").read_text())
    vtemplate = info_src.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    videos_linked = videos_copied = 0
    missing_videos = []
    for key, ci, fi in sorted(video_refs):
        rel = vtemplate.format(video_key=key, chunk_index=ci, file_index=fi)
        src, dst = source / rel, dest / rel
        if not src.exists():
            missing_videos.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.link(src, dst)
            videos_linked += 1
        except OSError:
            shutil.copy2(src, dst)
            videos_copied += 1
    if missing_videos:
        raise SystemExit(
            f"{len(missing_videos)} referenced video file(s) absent from source, e.g. {missing_videos[:3]}. "
            "A slice whose episodes point at missing videos loads and then fails on the first sample."
        )

    # meta/info.json -- counts updated, and the new column REGISTERED. An
    # unregistered column makes the dataset unloadable, the same trap `progress`
    # has: get_hf_features_from_features builds the schema from this dict.
    info = json.loads((source / "meta" / "info.json").read_text())
    info["total_episodes"] = len(ordered)
    info["total_frames"] = rows_out
    info["total_tasks"] = 1
    feats = info.setdefault("features", {})
    feats[GLOBAL_EP_COL] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
        "_comment": (
            "Episode index in the pristine combined corpus. episode_index here is "
            "renumbered densely because LeRobot v3 indexes episodes positionally; "
            "this column is what progress labels join on."
        ),
    }
    info["_sliced_from"] = {"task": task, "task_index": task_index, "source": str(source)}
    (dest / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    for extra in ("stats.json", "tasks.jsonl"):
        src = source / "meta" / extra
        if src.exists():
            shutil.copy2(src, dest / "meta" / extra)

    manifest = {
        "task": task,
        "task_index": task_index,
        "episodes": len(ordered),
        "frames": rows_out,
        "video_files": len(video_refs),
        "videos_linked": videos_linked,
        "videos_copied": videos_copied,
        "episode_map": {str(g): remap[g] for g in ordered},
        "info_sha256": _sha256(dest / "meta" / "info.json"),
        "tasks_sha256": _sha256(dest / "meta" / "tasks.parquet"),
    }
    (dest / "meta" / "slice_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="combined LeRobot v3 root")
    ap.add_argument("--task", required=True, help="task NAME, as in meta/tasks.parquet")
    ap.add_argument("--out", type=Path, required=True, help="parent dir; the slice lands at <out>/<task>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    m = slice_dataset(args.source, args.task, args.out, overwrite=args.overwrite)
    print(json.dumps({k: v for k, v in m.items() if k != "episode_map"}, indent=2))
    print(f"episodes {m['episodes']}, frames {m['frames']} -> {args.out / args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
