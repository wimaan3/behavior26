#!/usr/bin/env python3
"""Compute norm stats for a b1k config against an EXPLICIT dataset root.

WHY NOT UPSTREAM'S SCRIPT
-------------------------
`openpi/scripts/compute_norm_stats.py` takes only `--config-name`. It picks its
loader with

    elif data_config.dataset_root is not None:  -> create_b1k_dataloader(...)

so it does use the root-aware path -- but the root is whatever the CONFIG says,
and `pi05_b1k*` ships `dataset_root="./data/b1k/turning_on_radio"`. Pointed at a
slice that lives somewhere else, it would either 401 (local path missing -> Hub
fallback) or, worse, silently compute statistics over a stale dataset and write
them where training will read them. Computing the wrong thing over 1.25M frames
is expensive and invisible.

This mirrors upstream's `main()` exactly -- same transforms, same keys, same
output path (`config.assets_dirs / asset_id`) -- and only adds the overrides.

NOTE ON KEYS: upstream computes stats for `["state", "actions"]` only. `progress`
is NOT normalised, so both arms' files should be IDENTICAL, not merely equal on a
shared subset. `scripts/compare_norm_stats.py` asserts that.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys


import os

# ---------------------------------------------------------------- decode-free
# NORM_STATS_NO_DECODE=1 replaces LeRobot's video decode with zero frames of the
# right shape. Norm stats read only `state` and `actions`; decoding six camera
# streams per frame just to throw the pixels away measured at ~2 frames/s with 16
# workers (640 frames in 293 s on a 28-vCPU box), i.e. hours per arm.
#
# APPLIED AT MODULE IMPORT, not in main(): the DataLoader uses `spawn`, which
# re-imports this file in every worker as __mp_main__ but does NOT carry a
# monkeypatch made at runtime in the parent. An env var is inherited by spawned
# workers, so the patch lands everywhere or nowhere.
#
# NEVER TRUSTED ON FAITH: scripts/session_b/rung1.sh first computes the same
# seeded frames both ways and requires compare_norm_stats.py to report a
# bit-identical match before using this mode for the real pass.
if os.environ.get("NORM_STATS_NO_DECODE") == "1":
    try:
        import torch
        from lerobot.datasets import dataset_reader as _dr

        def _zero_frames(self, query_timestamps, ep_idx):
            out = {}
            for key, ts in query_timestamps.items():
                h, w, c = self._meta.features[key]["shape"]
                frame = torch.zeros((c, h, w), dtype=torch.float32)
                out[key] = frame if len(ts) == 1 else frame.unsqueeze(0).repeat(len(ts), 1, 1, 1)
            return out

        _dr.DatasetReader._query_videos = _zero_frames
    except Exception as e:  # noqa: BLE001 -- surface loudly, never silently decode
        raise SystemExit(f"NORM_STATS_NO_DECODE=1 but the decode patch failed to apply: {e}")


class RemoveStrings:
    """Drop string fields; JAX cannot hold them and stats do not need them.

    MODULE LEVEL, deliberately. With num_workers > 0 the DataLoader uses the
    `spawn` start method and must PICKLE every transform. A class defined inside
    main() is a local object and cannot be pickled -- the first run died with
    "Can't pickle local object 'main.<locals>.RemoveStrings'" the moment the
    workers started. Mirrors upstream, which also defines it at module level.
    """

    def __call__(self, x: dict) -> dict:
        import numpy as np
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True, nargs="+",
                    help="one task, or several (dataset_root is then their PARENT). Stats are "
                         "written under the FIRST repo_id, which is where openpi reads them.")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--assets-base-dir", required=True,
                    help="ABSOLUTE. openpi defaults to ./assets relative to cwd, so stats written "
                         "from one directory are silently skipped by training from another.")
    a = ap.parse_args()

    import numpy as np
    import tqdm
    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as transforms

    if not pathlib.Path(a.assets_base_dir).is_absolute():
        print(f"FAIL: --assets-base-dir {a.assets_base_dir!r} is relative")
        return 2
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from b1k_roots import RootError, apply_to_config, validate_roots
    try:
        plan = validate_roots(a.dataset_root, a.repo_id)
        cfg = apply_to_config(_config.get_config(a.config_name), plan, video_backend=a.video_backend)
    except RootError as e:
        print(f"FAIL: {e} -- refusing to compute stats over a stale path (does not exist or wrong layout)")
        return 2
    cfg = dataclasses.replace(cfg, assets_base_dir=a.assets_base_dir)

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = getattr(data_config, "dataset_root", None)
    print(f"config={a.config_name} root={root} repo_id={data_config.repo_id} "
          f"video_decode={'OFF (zero frames)' if os.environ.get('NORM_STATS_NO_DECODE') == '1' else 'on'}")
    if not root or not pathlib.Path(root).exists():
        print(f"FAIL: dataset_root {root!r} does not exist -- refusing to compute stats over a stale path")
        return 2
    if str(root) != str(plan.dataset_root):
        print(f"FAIL: config resolved to {root!r}, not the requested {str(plan.dataset_root)!r}")
        return 2

    dataset = _data_loader.create_b1k_dataset(data_config=data_config,
                                              action_horizon=cfg.model.action_horizon)
    dataset = _data_loader.TransformedDataset(dataset, [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ])
    n = len(dataset)
    if a.max_frames is not None and a.max_frames < n:
        num_batches, shuffle = a.max_frames // cfg.batch_size, True
    else:
        num_batches, shuffle = n // cfg.batch_size, False
    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=cfg.batch_size,
                                          num_workers=a.num_workers, shuffle=shuffle,
                                          num_batches=num_batches)
    keys = ["state", "actions"]
    stats = {k: normalize.RunningStats() for k in keys}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="stats"):
        for k in keys:
            stats[k].update(np.asarray(batch[k]))
    norm_stats = {k: s.get_statistics() for k, s in stats.items()}

    asset_id = data_config.asset_id or data_config.repo_id
    if isinstance(asset_id, list):
        asset_id = asset_id[0]
    out = cfg.assets_dirs / asset_id
    print(f"frames={n} batches={num_batches} -> {out}")
    normalize.save(out, norm_stats)
    print(f"NORM_STATS_OK {out}/norm_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
