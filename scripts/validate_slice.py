#!/usr/bin/env python3
"""Validate a per-task slice against the REAL LeRobot (wensi-ai/lerobot@release/b1k)
and the REAL Jetson sidecar, before any GPU time is spent on it.

Four checks, each printed as PASS/FAIL with the evidence:
  1. LeRobotDataset opens the slice: schema accepted, episode metadata shape right,
     video paths resolve -- and action chunks come from the RIGHT frames (the
     direct real-data check on the dataset_from_index / dataset_to_index trap).
  2. global_episode_index survives the round trip and differs from episode_index.
  3. Every task_index in the data resolves in the shipped tasks.parquet (pandas).
  4. merge_progress_labels.py against the real sidecar, in protocol mode
     (--drop-unlabelled), lands every label on the GLOBALLY correct episode --
     checked by value AND by the sidecar's own task_index. 4b: the merged root
     still opens in LeRobot.
Exit code is the number of failed checks.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: dict[str, str] = {}


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = "PASS" if ok else "FAIL"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def read_data(root: Path):
    import pandas as pd, pyarrow.parquet as pq
    return pd.concat([pq.read_table(p).to_pandas() for p in sorted((root / "data").glob("*/*.parquet"))],
                     ignore_index=True)


def open_dataset(root: Path, task: str, horizon: int, backend: str | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((root / "meta" / "info.json").read_text())
    kw = dict(repo_id=task, root=root,
              delta_timestamps={"action": [t / info["fps"] for t in range(horizon)]})
    if backend:
        kw["video_backend"] = backend
    return LeRobotDataset(**kw)


def check1(root: Path, task: str, horizon: int) -> None:
    import numpy as np
    raw = read_data(root)
    last_err = None
    for backend in (None, "pyav"):
        try:
            ds = open_dataset(root, task, horizon, backend)
            break
        except Exception as e:                       # noqa: BLE001
            last_err = e
    else:
        record("1 LeRobotDataset opens", False, f"{type(last_err).__name__}: {last_err}")
        traceback.print_exception(last_err)
        return
    meta = ds.meta
    record("1a schema + counts", len(ds) == len(raw) and meta.total_episodes == raw.episode_index.nunique(),
           f"len={len(ds)} rows={len(raw)} episodes meta={meta.total_episodes} data={raw.episode_index.nunique()} "
           f"backend={backend or 'default'}")

    missing = []
    for ep in range(meta.total_episodes):
        for vk in meta.video_keys:
            if not (root / meta.get_video_file_path(ep, vk)).exists():
                missing.append((ep, vk))
    record("1b video paths resolve", not missing,
           f"{meta.total_episodes} episodes x {len(meta.video_keys)} video keys, missing={missing[:3]}")

    # Action-chunk alignment on REAL data: item[i]['action'][k] must equal the raw
    # action row at index i+k, clamped to the episode. Probe start, middle, end of
    # first and last episode -- the boundaries are where a stale range shows.
    probes, bad = [], []
    eps = sorted(raw.episode_index.unique())
    for ep in (eps[0], eps[len(eps) // 2], eps[-1]):
        rows = raw[raw.episode_index == ep].sort_values("index")
        idxs = rows["index"].tolist()
        probes += [idxs[0], idxs[len(idxs) // 2], idxs[-1]]
    act = {int(r["index"]): np.asarray(r["action"], dtype=np.float32) for _, r in raw.iterrows()} \
        if len(raw) < 400_000 else None
    decoded_video = None
    for i in probes:
        item = ds[i]
        ep = int(item["episode_index"])
        rows = raw[raw.episode_index == ep]
        lo, hi = int(rows["index"].min()), int(rows["index"].max()) + 1
        got = item["action"].numpy()
        for k in range(horizon):
            j = max(lo, min(hi - 1, i + k))
            want = np.asarray(rows.loc[rows["index"] == j, "action"].iloc[0], dtype=np.float32)
            if not np.allclose(got[k], want, atol=1e-5):
                bad.append((i, k, j))
                break
        if decoded_video is None:
            vks = [k for k in item if k in meta.video_keys]
            decoded_video = {k: tuple(item[k].shape) for k in vks}
    record("1c action chunks from the right frames", not bad,
           f"{len(probes)} probes x horizon {horizon}; mismatches={bad[:3]}")
    record("1d a video frame decodes", bool(decoded_video), f"shapes={decoded_video}")
    return ds


def check2(root: Path, ds) -> None:
    raw = read_data(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    col = "global_episode_index"
    in_item = ds is not None and col in ds[0]
    per_ep = raw.groupby("episode_index")[col].nunique()
    record("2 global_episode_index survives",
           col in info["features"] and in_item and bool((per_ep == 1).all())
           and bool((raw[col] != raw["episode_index"]).any()),
           f"registered={col in info['features']} in LeRobot item={in_item} "
           f"one-global-per-local={bool((per_ep == 1).all())} "
           f"local {raw.episode_index.min()}..{raw.episode_index.max()} -> "
           f"global {raw[col].min()}..{raw[col].max()}")


def check3(root: Path, task: str) -> None:
    import pandas as pd
    raw = read_data(root)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    table = {int(v): str(k) for k, v in tasks["task_index"].items()}
    present = sorted(int(x) for x in raw.task_index.unique())
    unresolved = [t for t in present if t not in table]
    record("3 task_index resolves", not unresolved and all(table[t] == task for t in present),
           f"present={present} -> {[table.get(t) for t in present]}; table has {len(table)} tasks")


def check4(root: Path, task: str, labels: Path, work: Path, horizon: int) -> None:
    import numpy as np, pandas as pd
    out = work / "merged"
    cmd = [sys.executable, str(REPO / "scripts" / "merge_progress_labels.py"),
           "--dataset-root", str(root), "--labels", str(labels), "--out-root", str(out),
           "--join-on", "global_episode_index", "frame_index",
           "--labels-join-on", "episode_index", "frame_index", "--drop-unlabelled"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-1500:], res.stderr[-1500:])
    if res.returncode != 0:
        record("4 merge against real sidecar", False, f"merge exit {res.returncode}")
        return
    merged_root = out if (out / "data").exists() else out / task
    m = read_data(merged_root)
    side = pd.read_parquet(labels).set_index(["episode_index", "frame_index"])
    keys = pd.MultiIndex.from_arrays([m["global_episode_index"], m["frame_index"]])
    want = side["progress"].reindex(keys).to_numpy(dtype="float32")
    got = m["progress"].to_numpy(dtype="float32")
    val_bad = int((~np.isclose(got, want, atol=1e-6)).sum())
    task_bad = int((side["task_index"].reindex(keys).to_numpy() != m["task_index"].to_numpy()).sum())
    record("4 labels on globally correct episodes", val_bad == 0 and task_bad == 0 and len(m) > 0,
           f"rows={len(m)} value_mismatch={val_bad} sidecar_task_mismatch={task_bad} "
           f"episodes kept={m.episode_index.nunique()}")
    # 5 -- STRUCTURAL, checked without LeRobot so the failure is unambiguous.
    # A dropped-but-uncompacted root puts every later episode off by one
    # positionally, breaks the frame ranges, and sends LeRobot to the Hub for a
    # repo that does not exist -- a 401 that looks like an auth problem and is
    # really an off-by-N. Found exactly that way on 2026-09-14, so it is a
    # first-class check rather than a bonus.
    import pyarrow.parquet as pq
    eps_files = sorted((merged_root / "meta" / "episodes").glob("*/*.parquet"))
    eps = pd.concat([pq.read_table(f).to_pandas() for f in eps_files], ignore_index=True) if eps_files else None
    info = json.loads((merged_root / "meta" / "info.json").read_text())
    local = sorted(int(e) for e in m["episode_index"].unique())
    idx = m["index"].to_numpy()
    problems = []
    if local != list(range(len(local))):
        problems.append(f"episode_index not dense: gaps at {sorted(set(range(max(local) + 1)) - set(local))[:5]}")
    if sorted(idx.tolist()) != list(range(len(idx))):
        problems.append(f"index not dense: {idx.min()}..{idx.max()} for {len(idx)} rows")
    if eps is not None and len(eps) != len(local):
        problems.append(f"meta/episodes lists {len(eps)} episodes but data has {len(local)}")
    if info.get("total_episodes") != len(local):
        problems.append(f"info.json says {info.get('total_episodes')} episodes, data has {len(local)}")
    if eps is not None and {"dataset_from_index", "dataset_to_index"} <= set(eps.columns):
        for ep in eps.itertuples():
            rows = m[m["episode_index"] == ep.episode_index]["index"]
            if sorted(rows.tolist()) != list(range(ep.dataset_from_index, ep.dataset_to_index)):
                problems.append(f"episode {ep.episode_index} range does not match its rows")
                break
    record("5 merged root is compacted", not problems, "; ".join(problems) or "dense, pruned, ranges match")

    man = merged_root / "meta" / "progress_filter.json"
    mf = json.loads(man.read_text()) if man.exists() else {}
    record("5b filter manifest agrees with the root",
           bool(mf) and mf.get("episodes_out") == len(local) and mf.get("compacted") is True,
           f"manifest episodes_out={mf.get('episodes_out')} compacted={mf.get('compacted')} data={len(local)}")

    try:
        ds = open_dataset(merged_root, task, horizon, None)
        _ = ds[0]
        record("5c merged root opens in LeRobot", True, f"len={len(ds)} episodes={ds.meta.total_episodes}")
    except Exception as e:                           # noqa: BLE001
        record("5c merged root opens in LeRobot", False, f"{type(e).__name__}: {str(e)[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=16)
    a = ap.parse_args()
    ds = None
    for name, fn in (("1", lambda: check1(a.slice, a.task, a.horizon)),):
        try:
            ds = fn()
        except Exception as e:                       # noqa: BLE001
            record("1 LeRobotDataset opens", False, f"{type(e).__name__}: {e}"); traceback.print_exc()
    for fn in (lambda: check2(a.slice, ds), lambda: check3(a.slice, a.task),
               lambda: check4(a.slice, a.task, a.labels, a.work, a.horizon)):
        try:
            fn()
        except Exception as e:                       # noqa: BLE001
            record(getattr(fn, "__name__", "check"), False, f"{type(e).__name__}: {e}"); traceback.print_exc()
    print("\nSUMMARY", json.dumps(RESULTS))
    return sum(v == "FAIL" for v in RESULTS.values())


if __name__ == "__main__":
    sys.exit(main())
