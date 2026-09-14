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
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--progress-key", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--skip-norm-stats", action="store_true")
    a = ap.parse_args()

    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    cfg = _config.get_config(a.config)
    data = cfg.data
    base = dataclasses.replace(data.base_config, dataset_root=a.dataset_root, repo_id=a.repo_id)
    if a.progress_key:
        # The patch may hang progress_key off either the factory or the base
        # config depending on revision; set whichever exists rather than guess.
        if hasattr(base, "progress_key"):
            base = dataclasses.replace(base, progress_key=a.progress_key)
        elif hasattr(data, "progress_key"):
            data = dataclasses.replace(data, progress_key=a.progress_key)
        else:
            print("!! neither base_config nor data has progress_key -- patch not applied?")
            return 2
    data = dataclasses.replace(data, base_config=base, repo_id=a.repo_id)
    cfg = dataclasses.replace(cfg, data=data, batch_size=a.batch_size)

    print(f"config={a.config} repo_id={a.repo_id} root={a.dataset_root} "
          f"progress_key={a.progress_key} batch={a.batch_size}")
    loader = _data_loader.create_data_loader(
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
