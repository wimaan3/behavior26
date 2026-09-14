# Data pipeline verified end to end on real data — CPU pod, 2026-09-14

cpu3g 4 vCPU / 16 GB, EU-RO-1, $0.16/hr, 40 GB disk. Terminated. **Cost ≈ $0.07**
(~25 min). No GPU time. Real LeRobot at the pinned branch, real chunk-010, real
Jetson sidecar, real openpi at `0cc8e355` + our patches.

## All ten slice/merge checks PASS

```
1a schema + counts ....... len=1,253,243  meta 200 == data 200 episodes
1b video paths ........... 200 episodes x 6 video keys, none missing
1c action chunks ......... 9 probes x horizon 16 at episode boundaries, 0 mismatches
1d video decodes ......... rgb 3x480x480 / 3x720x720, depth 1x480x480
2  global_episode_index .. registered, in the LeRobot item, local 0..199 -> global 2000..2199
3  task_index ............ [10] -> set_up_a_coffee_station_in_your_kitchen, table has 100
4  labels globally correct  1,247,890 rows, 0 value mismatches, 0 sidecar-task mismatches
5  merged root compacted .. dense, pruned, ranges match
5b manifest agrees ....... episodes_out=199, compacted=True, data=199
5c opens in LeRobot ...... len=1,247,890  episodes=199
```

5 and 5c are the two that FAILED before compaction. Arm B's training root now
loads.

## Both arms pull a real batch through openpi's transform stack

`scripts/first_batch.py`, no model, no checkpoint:

| | arm A (`pi05_b1k_frozen_vlm`) | arm B (`..._progress`) |
|---|---|---|
| actions | (2, 32, 32) float32 | (2, 32, 32) float32 |
| state | (2, 32) float32 | (2, 32) float32 |
| images | 3 x (2, 224, 224, 3) + masks | same |
| **progress** | absent | **(2, 1) float32, range [0.1667, 0.1667]** |

So RepackTransform, **PromptFromLeRobotTask** (the task_index trap), B1KInputs,
the delta-action mapping and the tokenizer all accept the sliced+merged root, and
the progress column reaches the model's input on arm B and only on arm B.

0.1667 = 1/6 is a plausible first-frame progress on a D=6 task; both sampled
frames came from the same episode start, hence the single value.

## Four environment defects found, all in the probe or the box, none in the data

1. **`uv`'s wheel cache hit 16 GB** on a 40 GB box (two venvs of CUDA wheels) and
   the openpi install died with ENOSPC. CPU pods cap container disk at 40 GB;
   `UV_NO_CACHE=1` and dropping the LeRobot venv after validation fixed it.
2. **The probe called openpi's GENERIC loader.** `create_data_loader` routes to
   `create_torch_dataset`, which calls `LeRobotDatasetMetadata(repo_id)` with **no
   root** -- it ignores `dataset_root` entirely and goes to the Hub. The real
   trainer (`scripts/b1k/train_b1k.py:432`) uses `create_b1k_data_loader`. A probe
   that does not mirror the trainer proves nothing about the trainer; a test now
   pins the entry point.
3. **`libtorchcodec` needs system FFmpeg** the selkies image does not ship. PyAV
   bundles its own and openpi forwards `dataset_kwargs`, so `--video-backend pyav`
   is a supported knob rather than a patch.
4. **A crashing DataLoader worker reports only** `killed by signal: Terminated`,
   hiding the exception. `--num-workers 0` keeps decoding in-process so the real
   error surfaces.

## The 401 that means four different things

`RepositoryNotFoundError: 401` appeared **four** times today and never once meant
authentication:

* the uncompacted merged root (episode gap -> metadata load failed -> Hub fallback)
* a `dataset_root` that did not exist
* openpi's generic loader ignoring `dataset_root`
* and, with `HF_HUB_OFFLINE=1`, the same failure wearing `OfflineModeIsEnabled`

In every case the real cause was a local path or a wrong entry point. LeRobot
only reaches the Hub inside `except (FileNotFoundError, NotADirectoryError)`, so
**a 401 from this stack should be read as "local load failed", never as auth.**
`first_batch.py` now prints the effective `repo_id` / `dataset_root` / `asset_id`
and refuses a missing root before the Hub can mask it.

## What is still unproven before rung 1

* **Norm stats.** Every batch above ran `--skip-norm-stats`. Computing them is
  step 4 of the fork's b1k.md and must happen on the GPU box before training.
* **The 3.3B model and the ~7 GB checkpoint** -- deliberately untouched here.
* **Throughput.** Nothing about steps/sec is measured by a single batch.
