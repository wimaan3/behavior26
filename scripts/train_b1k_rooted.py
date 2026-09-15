#!/usr/bin/env python3
"""Run openpi's own b1k training loop against an EXPLICIT dataset root.

WHY THIS EXISTS -- three things openpi's CLI cannot express
------------------------------------------------------------
1. `DataConfigFactory.base_config` is `tyro.conf.Suppress[DataConfig | None]`.
   `--data.base_config.dataset_root` is therefore not a flag at all -- tyro
   rejects it at parse -- and every pi05_b1k config ships
   `dataset_root="./data/b1k/turning_on_radio"`. The root has to be set in code.
2. `assets_base_dir` defaults to "./assets", relative to the CURRENT DIRECTORY.
   Norm stats computed from one cwd are silently absent to training launched
   from another: openpi logs "Norm stats not found ... skipping" and trains
   UNNORMALISED. Pinned absolute here and in compute_norm_stats_b1k.py.
3. `log_interval` defaults to 100. Rung 3 needs action_loss and progress_loss at
   every one of the first 100 steps.

It calls `scripts/b1k/train_b1k.py::main` unchanged -- same data loader
(`create_b1k_data_loader`), same weight loader (so arm B exercises
`missing_regex=".*(lora|progress).*"` against the real pi05_base download), same
train_step, same checkpointing. Only the config it receives differs.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True, nargs="+",
                    help="one task (dataset_root IS the dataset) or several (dataset_root is their "
                         "PARENT; MultiLeRobotDataset). See scripts/b1k_roots.py.")
    ap.add_argument("--protocol", action="store_true",
                    help="an A/B arm: every task root must be merged, filtered and compacted")
    ap.add_argument("--assets-base-dir", required=True, help="ABSOLUTE; see module docstring")
    ap.add_argument("--checkpoint-base-dir", required=True)
    ap.add_argument("--num-train-steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--log-interval", type=int, default=1)
    ap.add_argument("--save-interval", type=int, default=10**9,
                    help="default: never mid-run. train_b1k still saves at the final step.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--progress-key", default=None)
    ap.add_argument("--progress-loss-weight", type=float, default=None)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--log-term-grads", action="store_true",
                    help="log per-term gradient norms for lambda calibration (2 extra backward passes/step)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", "/opt/openpi"))
    a = ap.parse_args()

    for flag, p in (("--assets-base-dir", a.assets_base_dir), ("--checkpoint-base-dir", a.checkpoint_base_dir)):
        if not pathlib.Path(p).is_absolute():
            print(f"FAIL: {flag} {p!r} is relative. A relative assets dir is how norm stats get "
                  f"written from one cwd and silently skipped by training from another.")
            return 2

    from openpi.training import config as _config

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from b1k_roots import RootError, apply_to_config, validate_roots

    try:
        plan = validate_roots(a.dataset_root, a.repo_id, progress_key=a.progress_key, protocol=a.protocol)
        cfg = apply_to_config(_config.get_config(a.config), plan,
                              video_backend=a.video_backend, progress_key=a.progress_key)
    except RootError as e:
        print(f"FAIL: {e}")
        return 2
    print(f"tasks             {plan.repo_ids} ({'MultiLeRobotDataset' if plan.multi else 'LeRobotDataset'})")

    model = cfg.model
    if a.progress_loss_weight is not None:
        if not hasattr(model, "progress_loss_weight"):
            print("FAIL: --progress-loss-weight given but the model config has no such field")
            return 2
        model = dataclasses.replace(model, progress_loss_weight=a.progress_loss_weight)

    cfg = dataclasses.replace(
        cfg, model=model, exp_name=a.exp_name,
        assets_base_dir=a.assets_base_dir, checkpoint_base_dir=a.checkpoint_base_dir,
        num_train_steps=a.num_train_steps, batch_size=a.batch_size,
        log_interval=a.log_interval, save_interval=a.save_interval,
        num_workers=a.num_workers, wandb_enabled=a.wandb, overwrite=a.overwrite,
    )
    if a.log_term_grads:
        if not hasattr(cfg, "log_loss_term_grad_norms"):
            print("FAIL: --log-term-grads but TrainConfig has no log_loss_term_grad_norms (patch 0002 out of date?)")
            return 2
        cfg = dataclasses.replace(cfg, log_loss_term_grad_norms=True)

    effective = cfg.data.create(cfg.assets_dirs, cfg.model)
    root = getattr(effective, "dataset_root", None)
    print(f"config            {cfg.name}")
    print(f"exp_name          {cfg.exp_name}")
    print(f"dataset_root      {root}")
    print(f"repo_id           {effective.repo_id}")
    print(f"assets_dir        {cfg.assets_dirs}")
    print(f"norm_stats loaded {effective.norm_stats is not None}")
    print(f"checkpoint_dir    {cfg.checkpoint_dir}")
    print(f"progress_key      {getattr(cfg.data, 'progress_key', None)}")
    print(f"progress_weight   {getattr(cfg.model, 'progress_loss_weight', None)}")
    print(f"weight_loader     {cfg.weight_loader}")
    print(f"term_grad_norms   {getattr(cfg, 'log_loss_term_grad_norms', None)}")
    if not root or not pathlib.Path(root).exists():
        print(f"FAIL: dataset_root {root!r} does not exist")
        return 2
    if str(root) != str(plan.dataset_root):
        print(f"FAIL: config resolved to {root!r}, not the requested {str(plan.dataset_root)!r}")
        return 2
    if effective.norm_stats is None:
        print(f"FAIL: no norm stats under {cfg.assets_dirs}. openpi would log 'skipping' and "
              f"train UNNORMALISED. Run scripts/compute_norm_stats_b1k.py with the same "
              f"--assets-base-dir first.")
        return 2

    train_py = pathlib.Path(a.openpi_root) / "scripts" / "b1k" / "train_b1k.py"
    spec = importlib.util.spec_from_file_location("train_b1k", train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(train_py.parent))
    spec.loader.exec_module(mod)
    mod.main(cfg)
    print("TRAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
