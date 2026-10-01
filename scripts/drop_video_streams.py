#!/usr/bin/env python3
"""Remove video streams the model never reads from a LeRobot v3 dataset root.

LeRobotDataset decodes EVERY video feature in meta/info.json for every sample. The
BEHAVIOR demos carry six streams per episode -- three RGB and three depth -- and openpi's
b1k robot config (src/openpi/configs/robots/b1k.py) maps only the three RGB ones. Measured
on the shot-one pod, 2026-10-01, 15 real samples: 4.09 s/sample decoding all six, 1.03 s
decoding RGB only. The depth streams were three-quarters of the decode cost, and decode
cost was the training stall that rung 4 blamed on worker count.

Removing a stream from `features` stops LeRobot decoding it; the frames the model does
read are untouched. The removed streams' video directories in THIS root are deleted
(in a merged root they are hard links, so the source copies remain).

    python scripts/drop_video_streams.py --root <task root> [--drop-prefix observation.depth]

Idempotent. Refuses to remove an RGB stream or to leave no video stream at all.
Writes meta/dropped_streams.json recording what it removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def plan(features: dict, prefixes: list[str]) -> tuple[list[str], list[str]]:
    video = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    drop = [k for k in video if any(k.startswith(p) for p in prefixes)]
    keep = [k for k in video if k not in drop]
    return drop, keep


def write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--drop-prefix", action="append", default=None,
                    help="feature-key prefix to drop (repeatable); default observation.depth")
    a = ap.parse_args()
    prefixes = a.drop_prefix or ["observation.depth"]

    root = Path(a.root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"FAIL: {info_path} not found")
        return 2
    info = json.loads(info_path.read_text())
    drop, keep = plan(info["features"], prefixes)

    if not keep:
        print(f"FAIL: dropping {drop} would leave no video stream")
        return 2
    rgb_dropped = [k for k in drop if ".rgb." in k]
    if rgb_dropped:
        print(f"FAIL: refusing to drop RGB stream(s) the model reads: {rgb_dropped}")
        return 2

    if drop:
        for k in drop:
            info["features"].pop(k)
        write_atomic(info_path, json.dumps(info, indent=4))
        for k in drop:
            d = root / "videos" / k
            if d.exists():
                shutil.rmtree(d)
    manifest = root / "meta" / "dropped_streams.json"
    prior = json.loads(manifest.read_text()) if manifest.exists() else {"dropped": []}
    record = {"prefixes": prefixes, "dropped": sorted(set(prior["dropped"]) | set(drop)),
              "kept_video_keys": keep}
    write_atomic(manifest, json.dumps(record, indent=2))
    print(f"DROPPED {len(drop)} stream(s) {drop or '(already dropped)'}; video streams kept: {keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
