# Session B, first attempt — blocked at rung 1 on dataset preparation

Pod: RTX PRO 4500 Blackwell (32 vCPU, 32 GB), EU-RO-1, $0.72/hr, 200 GB container
disk. Terminated. **Cost ≈ $0.55** (0.75 h). Nothing irreplaceable was on it.

## What cleared

| check | result |
|---|---|
| **JAX on sm_120 (Blackwell)** | **PASS** — `devices: [CudaDevice(id=0)]`, `backend: gpu`, compute cap 12.0, driver 580.126.09 |
| openpi install (uv sync) | 8.4 GB venv, ~2 min |
| all patches apply | 8 files, 413 insertions, clean |
| b1k configs register | `pi05_b1k`, `_frozen_vlm`, `_lora`, **`_frozen_vlm_progress`** |
| `tests/test_progress_head.py` vs patched openpi | **15 passed** on real JAX |
| `estimate_memory.py` | `_frozen_vlm_progress`: 3.353B total, 0.430B trainable, **13.5 G state** — identical to `_frozen_vlm`, so **the progress head costs no extra memory** |
| corpus download | chunk-010 + chunk-022, **30 GB total (15 GB/task)** in 95 s |
| task → chunk mapping | read from `meta/tasks.parquet`, not assumed: coffee station = 10, shoe rack = 22 |

The Blackwell/JAX question is settled: sm_120 works. Device enumeration is not
proof every kernel compiles — that needs a real step — but the gate passed and
no L4 fallback is required.

## What blocked it

**The config wants a PER-TASK LeRobot dataset; the challenge ships ONE combined
repo.**

```
pi05_b1k_frozen_vlm:  repo_id = turning_on_radio
                      base_config.dataset_root = ./data/b1k/turning_on_radio
```

but `behavior-1k/2026-challenge-demos` is a single 100-task LeRobot dataset —
`data/chunk-NNN/file-NNN.parquet` with one shared `meta/`. Checked: **no per-task
datasets are published** under `behavior-1k` (only the combined 2026/2025 demos,
rawdata, task-instances, robot-assets).

So slicing chunk-NNN into a standalone per-task dataset — its own `meta/info.json`,
episodes, tasks, stats, with episode and task indices renumbered — is **required
work we have not written**. It is step 1 of the fork's `docs/b1k.md` ("convert it
to LeRobot format before training"); our repo assumed the corpus was ready to use.

This is missing pipeline, not a bug, and it is the thing to build before the next
session-B pod.

## Four CLI/environment drifts found on the way

`scripts/train_cloud.sh` has never been run end to end against openpi `0cc8e355`.
Each of these cost a launch:

1. **`huggingface-cli download` is gone.** hub 1.x moved the CLI to `hf`. The old
   binary still exists and still runs, so it fails by printing help and exiting 1
   — it reads as a bad argument, not a renamed tool. Fixed in `download_data.sh`.
2. **`msgpack-numpy` missing.** Caught by `train_cloud.sh`'s own gate, which
   refuses to start a run whose checkpoint could not afterwards be served. The
   script working as designed.
3. **`compute_norm_stats.py` has no dataset-root flag** — only `--config-name`
   and `--max-frames`. `train_cloud.sh` passes `--data.base_config.dataset_root`,
   which that script never accepted at this commit. The root must come from the
   config, which is what exposed the layout mismatch above.
4. (From session A, same family) `serve_b1k.py` rejects the `policy:checkpoint`
   subcommand its own docs show.

## Measurement note for whoever runs rung 1

**Do not read peak VRAM off `nvidia-smi` with preallocation on.** JAX takes 90%
of the card at init, so the number is the flag, not the model. Rung 1 needs
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to measure; rungs 4-5 want it on, per
`scripts/train_cloud.sh` and `scripts/serve_baseline.sh`.

## Corpus sizing correction

**15 GB per task, not the ~33 GB we budgeted.** Eight tasks would be ~120 GB and
would have fit the 200 GB disk after all.
