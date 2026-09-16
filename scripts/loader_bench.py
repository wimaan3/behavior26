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


# Rung 4 stalled every 24 steps at 8 workers -- so the pipeline holds workers*3
# batches, not workers*2. Using 2 leaves a third of the queue inside the window we
# call "sustained", and the reported rate comes out high.
PREFETCH_FACTOR = 3

# A rate needs enough batches PAST the queue to be a rate and not a sample of two.
MIN_SUSTAINED_BATCHES = 30


def measure(next_batch, batches: int, batch_size: int, queue_depth: int, clock=time.time) -> dict:
    """Time `batches` pulls and report the rate the loader PRODUCES at.

    The queue is prefilled while the first batch is made, so the first
    `queue_depth` pulls come from memory and say nothing about the loader. They are
    discarded. If what remains is too short to be a rate, this returns None and says
    so rather than reporting the drain rate -- the failure that produced
    "777 items/s KEEPS UP" for a loader that sustains ~5.
    """
    t0 = clock()
    next_batch()                      # discard: worker spin-up + first-batch transforms
    warm = clock() - t0

    times = []
    for _ in range(batches - 1):
        t = clock()
        next_batch()
        times.append(clock() - t)

    sustained = times[queue_depth:]
    base = {"first_batch_s": warm, "queue_depth": queue_depth,
            "burst_batch_s": statistics.median(times[:queue_depth]) if times[:queue_depth] else None,
            "n_sustained": len(sustained)}
    if len(sustained) < MIN_SUSTAINED_BATCHES:
        return {**base, "items_per_s": None, "mean_batch_s": None, "median_batch_s": None,
                "p90_batch_s": None,
                "why": f"too short: {len(sustained)} batches past a {queue_depth}-deep queue, "
                       f"need {MIN_SUSTAINED_BATCHES}. Raise --batches to "
                       f"{queue_depth + MIN_SUSTAINED_BATCHES + 1}."}
    mean = statistics.fmean(sustained)
    return {**base, "mean_batch_s": mean, "median_batch_s": statistics.median(sustained),
            "p90_batch_s": sorted(sustained)[int(len(sustained) * 0.9)],
            "items_per_s": batch_size / mean if mean else None, "why": ""}


def bench_loader(cfg, workers: int, batches: int, batch_size: int) -> dict:
    import dataclasses

    from openpi.training import data_loader as _dl

    cfg = dataclasses.replace(cfg, num_workers=workers, batch_size=batch_size)
    loader = _dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=batches, skip_norm_stats=True)
    it = iter(loader)
    depth = max(1, workers) * PREFETCH_FACTOR
    return {"workers": workers, **measure(lambda: next(it), batches, batch_size, depth)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True, nargs="+")
    ap.add_argument("--config", default="pi05_b1k_frozen_vlm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=200,
                    help="must exceed workers*PREFETCH_FACTOR by MIN_SUSTAINED_BATCHES, or the "
                         "result is the prefetch queue draining and not a rate")
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
                if r["items_per_s"] is None:
                    print(f"  workers {w:>2}: NO RESULT -- {r['why']}")
                    continue
                verdict = "KEEPS UP" if r["items_per_s"] >= need else f"short by {need - r['items_per_s']:.1f}/s"
                print(f"  workers {r['workers']:>2}: sustained mean {r['mean_batch_s']:.2f}s/batch "
                      f"(median {r['median_batch_s']:.2f}, n={r['n_sustained']} past a "
                      f"{r['queue_depth']}-deep queue), first {r['first_batch_s']:.0f}s "
                      f"-> {r['items_per_s']:.1f} items/s  {verdict}")
            except Exception as e:  # noqa: BLE001
                print(f"  workers {w:>2}: FAILED {type(e).__name__}: {str(e)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
