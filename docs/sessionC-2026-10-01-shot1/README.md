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
