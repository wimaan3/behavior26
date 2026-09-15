# Session B, rung 1 — both arms load, fit at batch 32, and train 10 steps

Pod `tumg5i8ru9a6ei`: RTX PRO 4500 Blackwell (sm_120, 32,623 MiB), 28 vCPU, EU-RO-1,
driver 580.178.04, $0.72/hr, 200 GB disk. **Terminated. Rung cost ≈ $1.00; project
total $12.27.** Script: `scripts/session_b/rung1.sh`, behavior26 `e748f28`,
openpi `0cc8e355` + patches 0001/0002, jax 0.5.3, torch 2.7.1+cu128, lerobot 0.5.2.

## Stages, in the order they ran

| stage | result |
|---|---|
| 1 data | chunk-010 → slice → merge `--drop-unlabelled`: **199 episodes, manifest == info.json** |
| 2 one batch per arm | `FIRST_BATCH_OK` × 2 through `create_b1k_data_loader` |
| 3a decode-free proof | decoded vs zero-frame stats on 640 seeded frames: **identical file** |
| 3b norm stats | 128,000 seeded frames per arm, decode-free, 748 s / 752 s |
| 3c **arm parity** | **identical file: True — 8 shared keys, 0 mismatched** |
| 4 ten steps, arm A | fits at **batch 32**, `TRAIN_DONE`, 449 s (incl. base-checkpoint download) |
| 4 ten steps, arm B | fits at **batch 32**, `TRAIN_DONE`, 345 s |

### Norm stats fingerprint

```
9a3dcf3f7d5643913e581722133a7b878495ac19202d54e5d45ccfcd93e99d74  pi05_b1k_frozen_vlm
9a3dcf3f7d5643913e581722133a7b878495ac19202d54e5d45ccfcd93e99d74  pi05_b1k_frozen_vlm_progress
```

Upstream normalises only `state` and `actions`, so the arms' files are not merely
equal on a shared subset — they are the same file. Both committed here.

## What the ten steps show

### Arm A is healthy, and the arms are the same model up to the head

| step | A action_loss | B action_loss | B progress_loss | B loss |
|---|---|---|---|---|
| 0 | 0.8116 | 0.8116 | 0.7084 | 0.8825 |
| 1 | 1.0469 | 1.0469 | 0.6886 | 1.1158 |
| 4 | 1.1798 | 1.1797 | 0.6944 | 1.2491 |
| 9 | 1.1992 | 1.1994 | 0.7104 | 1.2705 |

Same seed, same batches, same init: **arm A's action_loss sequence reproduces arm
B's to four decimals**, diverging only in the fourth place from step 4. That is
the strongest arm-A regression check available at rung 1 — the control is the
treatment minus the head, and nothing else differs. The small late divergence is
expected: the progress head reads the action expert's features, so its gradient
reaches trainable shared parameters.

`loss = action_loss + 0.1 × progress_loss` checks exactly: 0.8116 + 0.07084 = 0.8825.

### The weight loader fix, against the real checkpoint

* arm A: `missing_regex='.*lora.*'` — loads.
* arm B: `missing_regex='.*(lora|progress).*'` — **loads**. Without the fix the
  fresh progress head fails `check_pytree_equality` at startup.

`pi05_base` restored from the real GCS download: **12.5 GiB** (not the ~7 GB
estimated), 112 s to download, 6.8 s to restore; cached for the second arm.

### progress_loss sits at ln 2

0.689–0.710 across all ten steps, i.e. ≈ 0.693 = ln 2: binary cross-entropy at an
uninformative p = 0.5. That is the correct starting point for a freshly initialised
head. It is **not yet evidence the head can learn** — see the warmup note below.

## Three findings that change the next rungs

### 1. The learning rate is ~0 for all of rung 1 and most of rung 2

The frozen-VLM configs inherit openpi's default `CosineDecaySchedule`:
**warmup_steps = 1,000, peak_lr = 2.5e-5**. At step 10 the rate is ~2.5e-7 — which
is why `param_norm` is flat at 1801.9884 — and at step 100 it is still only ~10%
of peak. Summed over the first 100 steps that is about **five peak-rate steps**.

So rung 2's manipulation check ("is progress_loss decreasing within 100 steps")
is at real risk of a **false negative**: a head that can learn perfectly well may
not visibly leave ln 2 on that budget. This is a pre-registration question, not a
bug, and it is raised for decision rather than changed here.

### 2. Peak VRAM here is the allocator ceiling, not the model's need

Both arms peaked at ~31.2 GB (31,218 / 31,206 MiB) — almost exactly
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` of the card plus the CUDA context. With
preallocation off, JAX's BFC arena still grows and does not shrink, so the reading
is where the arena stopped, not what the step required. **"Fits at batch 32" is
established; the headroom is not.** `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
allocates exactly and would give the true figure, at some speed cost.

### 3. Norm stats were parent-process-bound, not decode-bound

Video decode turned out not to be the constraint: raw PyAV random access ran at
~180 frames/s single-threaded, and LeRobot `__getitem__` at 56 ms with decode /
9 ms without. The openpi loader's parent process sat pinned at ~106% CPU while
workers idled near 60%, capping decode-free stats at ~170 frames/s regardless of
worker count. The same parent-side ceiling applies to the training loader, so
**rung 4's steps/sec may be data-bound**; that is the thing to watch.

torchcodec also works on this image once pointed at the FFmpeg 7 libraries PyAV
bundles (symlinks to the hashed sonames plus `av.libs` on `LD_LIBRARY_PATH`) —
a fallback if PyAV proves slow under training load.

## Other numbers worth keeping

* Checkpoint at step 9: **8.8 GB** per arm (estimate was ~13 GB).
* Disk after both runs: 56 G of 200 G.

## Defects fixed getting here (all pushed)

`RemoveStrings` defined inside `main()` was unpicklable by spawned workers; a
diagnostic rerun left a 64-frame norm-stats file that the launcher's
"norm stats loaded" check would have accepted (deleted before relaunch); and the
local test gate was not gating, because the Bash tool runs zsh, where
`${PIPESTATUS[0]}` is always empty.
