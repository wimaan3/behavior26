# Session C: shot one, 1 Oct 2026

Both A/B arms trained for real: coffee + shoes, 10,000 steps each, λ 0.15, budget $25.
Plan: `~/.claude/plans/dynamic-foraging-ember.md`. Live log lines are copied here at each milestone.

## Hardware, chosen on measurements

| | |
|---|---|
| Pod | `id5ngeyyeldrjk`, RTX PRO 4500 Blackwell 32 GB, EU-RO-1, $0.72/h |
| CPU / RAM | RunPod lists 28 vCPU; `nproc` reports **23** usable. 251 GB RAM |
| Container disk | 200 GB (training corpus ~33 GB/task) |
| Volume | `96mu3d0s32`, grown 150 → **200 GB**: 141 GB were already used, so only 9 GB would have been free |

24 GB cards are excluded: arm B peaks at 24.7 GB. The RTX PRO 6000 costs 2.9× as much,
and the loader couldn't feed even the 4500.

## Three things found before training started

1. **The volume check used `df`.** On a network volume, `df` reports the whole cluster
   (439 TB). It would have passed at 9 GB free, and training would have died at its first
   checkpoint. It now uses `du` against the quota. Measured: 141 GB used.
2. **The LR schedule decayed over a fixed 30,000 steps.** A 10k run would have ended
   un-annealed. Decay now follows `STEPS` (`decay_steps=10000` confirmed in the arm A log).
3. **The pod cannot stop itself.** RunPod's injected per-pod key returns 403 for get/stop/
   terminate on its own pod, with runpodctl v1.14.15 and v2.14.0 alike. The runner now writes
   `TERMINAL` to `STATUS`, and `scripts/watchdog.sh` on the owner's laptop terminates the pod.

## Attempt 1: the stall was never worker count

Arm A at 20 workers (attempt 1, 00:32–01:00 UTC, stopped at step 193):

| | 8 workers (rung 4) | 20 workers (attempt 1) |
|---|---|---|
| mean s/step | 8.11 | **7.14** |
| non-stall step | 3.90 | 3.78 |
| stall pattern | every 24 steps, ~103 s | every 10–20 steps, 25–33 s |
| share of clock in stalls | 52.8% | 31.6% |

Each dataloader worker ran at ~110% CPU, so the loader was **CPU-bound on video decode**.
More workers can't fix that.

**The cause:** each demo carries six video streams, three RGB and three **depth**.
openpi builds a plain `LeRobotDataset`, which decodes every video feature for every sample.
openpi's b1k robot config maps only the three RGB streams, and nothing reads depth.
Measured on the pod, 15 real samples, single process:

| streams decoded | s/sample |
|---|---|
| all six | **4.09** |
| RGB only | **1.03** |

So three-quarters of the decode work produced frames that were thrown away. Rung 4's
"prefetch sawtooth" had the same cause; its fix (more workers) treated a symptom.

**Fix:** `scripts/drop_video_streams.py` removes the depth features from each task root.
shot one then checks the real dataset: LeRobot's `video_keys` must equal exactly the three RGB keys the
robot config maps, and a sample must load. Both tasks passed (`STREAMS_OK`).
The model's inputs are unchanged.

Attempt 1's logs are on the volume at `/workspace/shot1/attempt1-depth-decode/`.

## Attempt 2

Relaunched 01:07 UTC on commit `6d686be`, arm A training from 01:10. Results follow below.

### Step-300 gate: GO

```
GATE GO mean 3.68 s/step over steps 50..300 <= 4.50
```

| | rung 4 (8 workers, depth decoded) | attempt 1 (20 workers, depth decoded) | **attempt 2 (20 workers, RGB only)** |
|---|---|---|---|
| mean s/step | 8.11 | 7.14 | **3.68** |

Steady at 3.67–3.68 s/step from step 68 to 345, with no stalls. At this rate each arm takes
~10.3 h; both arms finish ~22:30 UTC on 1 Oct, ≈ $16 of pod time, inside the 26 h cap.
Next: the resume test at arm A's first checkpoint (step 1000), and the health gate.

### Health gate at step 1000: PASS

```
HEALTH PASS arm A: action_loss 0.7042 -> 0.1695 (z = 15.4)
```

### The resume test found a deadlock, and the run sat idle for 14.5 hours

At 02:15:31 UTC the supervisor killed the trainer once, as designed, right after
checkpoint 1000 was finalised. The python died, but **its multiprocessing helpers (two
dataloader workers and the resource tracker) survived**. They were re-parented to init and
kept the trainer's stdout pipe open. The timestamping `while read` sat in `pipe_read`
waiting for an EOF that never came, so `run_arm` never reached its retry. The GPU idled at
0% until 16:49 UTC. Nothing noticed: the laptop slept, the laptop watchdog had no key yet,
and the pod had no stall detection.

The runner's control-flow tests used a fake trainer with no children, so they couldn't
catch this.

**Cost:** pod `id5ngeyyeldrjk` total **$11.81**, of which ~$10.70 was the idle 14.5 h.
Arm A's step-1000 checkpoint is safe on the volume.

**Fix**, with executed tests that use a trainer that does spawn a child holding the pipe:
- The trainer runs as its own process group (`exec setsid`, so pid = pgid, recorded).
  `kill_trainer` kills the whole group. Every kill in the supervisor now uses it.
- `reap_orphans`: if the trainer dies on its own but its group lives on, kill the group.
- Stall killer on the pod: no trainer output for `STALL_MINUTES` (30) → kill the group,
  and `run_arm` auto-resumes from the last checkpoint.
- Laptop watchdog backstop: no training output for 90 min → terminate the pod.
- The control test, killing only the leader, reproduces the hang locally, so the fix's
  test is known to exercise the real bug.

The signal trap added earlier worked: the stopped run recorded `KILLED by signal`, then
`TERMINAL`.

## Attempt 3: relaunch on a fresh pod, resuming arm A

Pod `5h33bvuxn9tx74` (RTX PRO 4500; 23 usable vCPUs, so 20 workers; 251 GB RAM). It was
requested with `minVcpuCountPerGpu: 16` because stock for 24 was gone. Launched 19:24 UTC
on `031923d`, `STEPS=10000 MAX_HOURS=23`.

- Volume: 150 GB used of 200 (the step-1000 checkpoint is ~9 GB, smaller than the 16 GB
  estimate). Peak during arm B is ~168 GB.
- Data rebuilt on the new pod; **fingerprint `8a7853e2ea95d1ca cf9b6e5ec1b01a79` matches**
  the run the checkpoint came from. Depth dropped, `STREAMS_OK` on both tasks.
- Arm A started with `--resume`.

### Resume test: PASS, across pods

```
RESUME_TEST PASS: restarted at step 1001 (checkpoint 1000)
```

The first step logged on the new pod is 1001, so training continued from the volume's
checkpoint rather than restarting. Back at ~3.7 s/step. Arm A should finish ~04:50 UTC on
2 Oct, and arm B ~15:10 UTC.

## Result: both arms trained to 10,000 steps

Arm A finished 05:13 UTC, arm B 16:16 UTC on 2 Oct. The laptop watchdog saw `TERMINAL`
and terminated the pod at 16:21:34, five minutes later. Logs are in [`logs/`](logs/);
the manifest is [`logs/manifest.json`](logs/manifest.json).

| | Arm A (control) | Arm B (progress head, λ 0.15) |
|---|---|---|
| Config | `pi05_b1k_frozen_vlm` | `pi05_b1k_frozen_vlm_progress` |
| Steps, non-finite values | 10,000, 0 | 10,000, 0 |
| Health at step 1000 | action 0.704 → 0.170 (z = 15.4) | action 0.705 → 0.170 (z = 15.4); **progress 0.694 → 0.621 (z = 21.1)** |
| Action loss, last 1000 (mean / median) | 0.1147 / 0.0965 | 0.1144 / 0.0984 |
| Progress loss, last 1000 | — | **0.563** (chance: ln 2 = 0.693) |

Shared by both arms: data fingerprint `8a7853e2ea95d1ca cf9b6e5ec1b01a79`, norm stats
`d9dd9b98d20dda32`, seed 42, batch 32, LR decay over 10,000 steps, openpi `0cc8e35`.

### What the training logs do and don't say

- **The progress head learned.** Its loss fell well below chance and plateaued around step 7000.
- **Action loss, end of training:** B − A = −0.0003, 95% CI [−0.0060, +0.0053] (last
  1000 steps, unpaired). No measurable difference.
- **Action loss, first 1000 steps, exactly paired** (same seed, identical batches): B − A =
  **+0.0009, 95% CI [+0.0004, +0.0014]**. By the rule pre-registered on 2026-09-15
  ("degraded = interval entirely above 0"), that window **counts as degraded**. It is ~0.5% of
  the loss, and it is not distinguishable by the end. It's reported here as the rule
  requires, not explained away.
- After arm A's resume at step 1000, its data order restarted, so later steps are compared
  as windows of the same distribution, not batch-for-batch.
- **Training loss cannot say whether the head improves task success.** Only evaluation in
  the simulator can.

### Backups: three copies of the trained models

| Copy | Where | Verified |
|---|---|---|
| 1 | RunPod volume `96mu3d0s32`, `/workspace/shot1/checkpoints/` | — |
| 2 | Laptop `~/behavior26-checkpoints/arm{A,B}/9999` (8.8 GB each) | sha256 of every file matches the pod: A 29/29, B 30/30 |
| 3 | Per-file checksums in git: [`logs/checkpoint_armA_9999.sha256`](logs/checkpoint_armA_9999.sha256), [`armB`](logs/checkpoint_armB_9999.sha256) | — |

Copied with `runpodctl send/receive` through a $0.06/h CPU pod. CPU pods have no direct
SSH. One send failed on a transient TLS timeout fetching croc's relay list and succeeded
on retry. Each checkpoint carries its own `assets/…/norm_stats.json`, which
`serve_baseline.sh` requires before serving.

### Cost

Since the 30 Sep top-up: $29.08 to the end of training, plus ~$0.06 for the pull pod.
