#!/usr/bin/env python3
"""Pull ONE batch through openpi's real transform stack. No model, no checkpoint.

WHY THIS RUNS BEFORE THE MODEL
------------------------------
`validate_slice.py` proves LeRobot opens the root. It does NOT prove openpi
accepts it: RepackTransform, PromptFromLeRobotTask, B1KInputs, the delta-action
mapping and the tokenizer all sit between the dataset and the model, and
`PromptFromLeRobotTask` is exactly where the task_index trap lives --

    task_index = int(data["task_index"])
    if (prompt := self.tasks.get(task_index)) is None: raise ValueError(...)

Loading a 3.3B model and downloading a ~7 GB checkpoint before discovering a
KeyError in a transform means paying for the download to learn something one
batch answers in seconds. So: batch first, model second.

`--skip-norm-stats` keeps this runnable before norm stats exist (and on a CPU
box): it exercises every transform EXCEPT normalisation. Normalisation is the
one step that cannot silently mis-key -- it either finds its asset or it does
not -- so it is the cheapest thing to defer, not the riskiest.

    python scripts/first_batch.py --config pi05_b1k_frozen_vlm \\
        --dataset-root /opt/merged --repo-id set_up_a_coffee_station_in_your_kitchen
    python scripts/first_batch.py --config pi05_b1k_frozen_vlm_progress ... --progress-key progress
"""
from __future__ import annotations

import argparse
import dataclasses
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True, nargs="+")
    ap.add_argument("--progress-key", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--skip-norm-stats", action="store_true")
    ap.add_argument("--video-backend", default=None,
                    help="'pyav' avoids torchcodec, which needs system FFmpeg libs the "
                         "selkies image does not ship ('Could not load libtorchcodec'). "
                         "PyAV bundles its own. openpi forwards dataset_kwargs to LeRobot.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 keeps decoding in-process. A crashing worker is reported as "
                         "'DataLoader worker killed by signal: Terminated', which hides "
                         "whatever actually raised.")
    a = ap.parse_args()

    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    import pathlib as _pl0
    sys.path.insert(0, str(_pl0.Path(__file__).resolve().parent))
    from b1k_roots import RootError, apply_to_config, validate_roots
    try:
        plan = validate_roots(a.dataset_root, a.repo_id, progress_key=a.progress_key)
        cfg = apply_to_config(_config.get_config(a.config), plan,
                              video_backend=a.video_backend, progress_key=a.progress_key)
    except RootError as err:
        print(f"FAIL: {err}")
        return 2
    cfg = dataclasses.replace(cfg, batch_size=a.batch_size, num_workers=a.num_workers)

    print(f"config={a.config} repo_id={a.repo_id} root={a.dataset_root} "
          f"progress_key={a.progress_key} batch={a.batch_size}")

    # Print what the loader will ACTUALLY use, and refuse a root that is not
    # there. Every "local path missing" in this stack surfaces as an HF
    # RepositoryNotFoundError 401 -- the loader falls back to the Hub and the
    # error names auth rather than the path. Seen three times now: the merged
    # root, the uncompacted root, and here.
    import pathlib as _pl
    effective = cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f"  effective repo_id      {effective.repo_id}")
    print(f"  effective dataset_root {getattr(effective, 'dataset_root', None)}")
    print(f"  effective asset_id     {getattr(effective, 'asset_id', None)}")
    print(f"  norm_stats loaded      {getattr(effective, 'norm_stats', None) is not None}")
    _root = getattr(effective, "dataset_root", None)
    if _root and not _pl.Path(_root).exists():
        print(f"FAIL: effective dataset_root {_root} does not exist. The loader would "
              f"fall back to the Hub and report a 401 that has nothing to do with auth.")
        return 2
    # create_b1k_data_loader, NOT create_data_loader. `scripts/b1k/train_b1k.py:432`
    # uses the b1k loader; the generic one routes to create_torch_dataset, which
    # calls LeRobotDatasetMetadata(repo_id) with NO root -- it ignores
    # dataset_root entirely and goes to the Hub, producing a 401 that looks like
    # an auth failure and is really "you called the wrong entry point". A probe
    # that does not mirror the trainer proves nothing about the trainer.
    loader = _data_loader.create_b1k_data_loader(
        cfg, shuffle=False, num_batches=1, skip_norm_stats=a.skip_norm_stats)
    obs, actions = next(iter(loader))

    def describe(x):
        return f"{tuple(getattr(x, 'shape', ()))}:{getattr(x, 'dtype', type(x).__name__)}"

    print("BATCH OK")
    print("  actions      ", describe(actions))
    d = obs.to_dict() if hasattr(obs, "to_dict") else vars(obs)
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}.{kk:<26}", describe(vv))
        else:
            print(f"  {k:<28}", describe(v))
    prog = d.get("progress")
    if a.progress_key:
        if prog is None:
            print("FAIL: --progress-key given but the batch carries no progress")
            return 1
        import numpy as np
        arr = np.asarray(prog)
        print(f"  progress range  [{arr.min():.4f}, {arr.max():.4f}]")
    print("FIRST_BATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
