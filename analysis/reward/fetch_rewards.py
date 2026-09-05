#!/usr/bin/env python3
"""Build a reward-only mirror of the BEHAVIOR-1K LeRobot corpus.

The committed corpus stats were measured on the 21 task shards held locally
(5,677 of 20,000 episodes, 28.4%). Measuring `D` and `phi0` on all 100 tasks needs
`next.reward` for all 20,000 episodes -- but not the observations. Pulling `data/*`
whole is ~70 GB across 955 parquet files, of which the reward columns are 0.006%:

    observation.state          64.9%
    action                     12.1%
    observation.robot2cam_pose 22.1%
    next.reward                0.0018%   (1,511 bytes of 82 MB)

Parquet is columnar, so a projected read over HTTP range requests transfers only the
column chunks asked for. This writes each file's reward columns to a mirror that keeps
the LeRobot layout, so `labels.Dataset`, `measure_corpus` and `labels.py` run against
it **unmodified**.

`index` is not fetched -- it is the largest of the small columns and is exactly
`dataset_from_index + frame_index` from `meta/episodes/` (asserted per file below), so
it is synthesised rather than transferred.

Usage:
    python -m analysis.reward.fetch_rewards --data-root ~/behavior-data \
        --out ~/behavior-rewards
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import pathlib
import shutil
import sys
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "behavior-1k/2026-challenge-demos"

# What we actually pull over the wire. `index` is synthesised, not fetched.
FETCH_COLUMNS = ["episode_index", "frame_index",
                 "next.reward", "next.terminated", "next.truncated"]
OUT_COLUMNS = ["index"] + FETCH_COLUMNS

_tls = threading.local()


def _fs():
    """One HfFileSystem per thread -- the object is not documented as thread-safe."""
    if not hasattr(_tls, "fs"):
        from huggingface_hub import HfFileSystem
        _tls.fs = HfFileSystem()
    return _tls.fs


def episode_map(data_root: pathlib.Path) -> pd.DataFrame:
    """episode -> task, length, data shard, and global row offset, for all 20,000."""
    files = sorted(glob.glob(str(data_root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no meta/episodes under {data_root}")
    cols = ["episode_index", "task_index", "length", "data/chunk_index",
            "data/file_index", "dataset_from_index"]
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files],
                     ignore_index=True)


def _read_one(chunk: int, fidx: int, data_root: pathlib.Path) -> pa.Table:
    """Reward columns for one data shard: from disk when held, else over HTTP."""
    rel = f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet"
    local = data_root / rel
    if local.exists():
        return pq.read_table(local, columns=FETCH_COLUMNS)
    with _fs().open(f"datasets/{REPO}/{rel}", "rb") as fh:
        return pq.ParquetFile(fh).read(columns=FETCH_COLUMNS)


def build_shard(chunk: int, fidx: int, data_root: pathlib.Path, out_root: pathlib.Path,
                meta: pd.DataFrame) -> dict:
    """Mirror one shard. Returns a per-shard record; raises on any integrity failure."""
    dest = out_root / "data" / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.parquet"
    expect = meta[(meta["data/chunk_index"] == chunk) & (meta["data/file_index"] == fidx)]
    n_expected = int(expect.length.sum())

    if dest.exists() and pq.ParquetFile(dest).metadata.num_rows == n_expected:
        return {"chunk": chunk, "file": fidx, "rows": n_expected, "source": "cached"}

    source = "local" if (data_root / "data" / f"chunk-{chunk:03d}" /
                         f"file-{fidx:03d}.parquet").exists() else "hub"
    tb = _read_one(chunk, fidx, data_root)
    df = tb.to_pandas()

    # Integrity: the shard must hold exactly the episodes meta says it does, at the
    # lengths meta says, in order. If any of that is false the synthesised `index`
    # would be wrong, and a wrong join key is worse than no labels.
    if len(df) != n_expected:
        raise ValueError(f"{dest}: {len(df)} rows, meta says {n_expected}")
    got = df.groupby("episode_index", sort=True).size()
    want = expect.set_index("episode_index").length.sort_index()
    if not got.index.equals(want.index) or not (got.values == want.values).all():
        raise ValueError(f"{dest}: episode/length disagreement with meta")
    if not df.equals(df.sort_values(["episode_index", "frame_index"],
                                    kind="stable").reset_index(drop=True)):
        raise ValueError(f"{dest}: rows are not in (episode_index, frame_index) order")

    # index == dataset_from_index + frame_index, verified against the real column on
    # every locally-held shard by tests/test_labels.py.
    off = expect.set_index("episode_index").dataset_from_index
    df.insert(0, "index", off.reindex(df.episode_index).to_numpy() + df.frame_index.to_numpy())

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df[OUT_COLUMNS], preserve_index=False), tmp,
                   compression="zstd", compression_level=9)
    os.replace(tmp, dest)
    return {"chunk": chunk, "file": fidx, "rows": len(df), "source": source}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="local pull (needs meta/ at least)")
    p.add_argument("--out", required=True, help="mirror root to write")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    data_root = pathlib.Path(a.data_root).expanduser()
    out_root = pathlib.Path(a.out).expanduser()
    meta = episode_map(data_root)

    # meta/ is what makes the mirror self-describing; copy it once.
    if not (out_root / "meta" / "info.json").exists():
        (out_root).mkdir(parents=True, exist_ok=True)
        shutil.copytree(data_root / "meta", out_root / "meta", dirs_exist_ok=True)

    shards = sorted(map(tuple, meta[["data/chunk_index", "data/file_index"]]
                        .drop_duplicates().to_numpy().tolist()))
    print(f"{len(shards)} shards, {len(meta)} episodes, {meta.task_index.nunique()} tasks",
          flush=True)

    done, failed = [], []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_shard, int(c), int(f), data_root, out_root, meta): (c, f)
                for c, f in shards}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                done.append(fut.result())
            except Exception as e:                     # noqa: BLE001 -- reported, not hidden
                failed.append((futs[fut], repr(e)))
                print(f"FAIL chunk-{futs[fut][0]:03d}/file-{futs[fut][1]:03d}: {e}", flush=True)
            if i % 50 == 0 or i == len(shards):
                print(f"  {i}/{len(shards)} shards", flush=True)

    rows = sum(d["rows"] for d in done)
    from_hub = sum(1 for d in done if d["source"] == "hub")
    size = sum(f.stat().st_size for f in out_root.rglob("data/**/*.parquet"))
    print(f"\n{len(done)} shards ok ({from_hub} fetched from hub), {len(failed)} failed")
    print(f"{rows:,} frames mirrored, {size/1e6:.0f} MB on disk")
    if rows != int(meta.length.sum()):
        print(f"WARNING: {rows:,} frames != {int(meta.length.sum()):,} in meta")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
