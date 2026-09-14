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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--video-backend", default="pyav")
    a = ap.parse_args()

    import numpy as np
    import tqdm
    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as transforms

    class RemoveStrings(transforms.DataTransformFn):
        def __call__(self, x: dict) -> dict:
            return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

    cfg = _config.get_config(a.config_name)
    base = dataclasses.replace(cfg.data.base_config, dataset_root=a.dataset_root, repo_id=a.repo_id)
    kwargs = dict(getattr(base, "dataset_kwargs", None) or {})
    if a.video_backend:
        kwargs["video_backend"] = a.video_backend
    base = dataclasses.replace(base, dataset_kwargs=kwargs)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, base_config=base, repo_id=a.repo_id))

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = getattr(data_config, "dataset_root", None)
    print(f"config={a.config_name} root={root} repo_id={data_config.repo_id}")
    if not root or not pathlib.Path(root).exists():
        print(f"FAIL: dataset_root {root!r} does not exist -- refusing to compute stats over a stale path")
        return 2
    if str(root) != str(a.dataset_root):
        print(f"FAIL: config resolved to {root!r}, not the requested {a.dataset_root!r}")
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
