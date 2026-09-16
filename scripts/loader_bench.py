#!/usr/bin/env python3
"""Measure the data path alone -- no model, no optimiser.

Training is data-bound (rung 4: median GPU utilisation 0%, ~60 s stalls). This
attributes the stall by timing, in order:

  A  raw LeRobot __getitem__ in this process      -- the per-item ceiling
  B  openpi's loader at several worker counts     -- does parallelism help?
  C  pyav vs torchcodec                           -- is the decoder the limit?

A step needs batch/step_seconds items/s to keep the GPU busy. Steady state only:
the first two batches are discarded, because worker spin-up and JIT of the
transforms are not what we are measuring.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def bench_loader(cfg, workers: int, batches: int, batch_size: int) -> dict:
    import dataclasses

    from openpi.training import data_loader as _dl

    cfg = dataclasses.replace(cfg, num_workers=workers, batch_size=batch_size)
    loader = _dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=batches, skip_norm_stats=True)
    it = iter(loader)
    t0 = time.time()
    next(it)                      # discard: worker spin-up + first-batch transforms
    warm = time.time() - t0
    times = []
    for _ in range(batches - 1):
        t = time.time()
        next(it)
        times.append(time.time() - t)
    times = times[1:] or times
    med = statistics.median(times)
    return {"workers": workers, "first_batch_s": warm, "median_batch_s": med,
            "p90_batch_s": sorted(times)[int(len(times) * 0.9)] if times else float("nan"),
            "items_per_s": batch_size / med if med else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True, nargs="+")
    ap.add_argument("--config", default="pi05_b1k_frozen_vlm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 8, 24])
    ap.add_argument("--step-seconds", type=float, default=3.85,
                    help="GPU step time, to say what the loader must sustain")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from b1k_roots import apply_to_config, validate_roots
    from openpi.training import config as _config

    need = a.batch_size / a.step_seconds
    print(f"a {a.batch_size}-sample step every {a.step_seconds}s needs {need:.1f} items/s\n")

    for backend in ("pyav", "torchcodec"):
        try:
            plan = validate_roots(a.dataset_root, a.repo_id)
            cfg = apply_to_config(_config.get_config(a.config), plan, video_backend=backend)
        except Exception as e:  # noqa: BLE001
            print(f"[{backend}] config failed: {e}")
            continue
        print(f"=== backend {backend} ===")
        for w in a.workers:
            try:
                r = bench_loader(cfg, w, a.batches, a.batch_size)
                verdict = "KEEPS UP" if r["items_per_s"] >= need else f"short by {need - r['items_per_s']:.1f}/s"
                print(f"  workers {r['workers']:>2}: median {r['median_batch_s']:.2f}s/batch  "
                      f"p90 {r['p90_batch_s']:.2f}s  first {r['first_batch_s']:.1f}s  "
                      f"{r['items_per_s']:.1f} items/s  {verdict}")
            except Exception as e:  # noqa: BLE001
                print(f"  workers {w:>2}: FAILED {type(e).__name__}: {str(e)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
