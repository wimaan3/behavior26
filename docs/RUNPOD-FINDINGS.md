# What we learned from all the RunPod work (13–18 Sep 2026)

One-page summary of every finding from the rented-GPU sessions. Each line links to the
dated folder holding its evidence (logs, reports, commits). Written 2026-09-18.

**About 21 pods, $25.00 total** ($23.61 compute, $1.39 network-volume storage). The
account balance ran out on 2026-09-18; see [the end of this page](#where-things-stand).

## Evaluation: simulator and scoring (session A, 13–14 Sep)

Evidence: [`sessionA-2026-09-13/`](sessionA-2026-09-13/), `AB_PROTOCOL.md` revisions 13e–13g and 2026-09-14.

| Finding | Value | Why it matters |
|---|---|---|
| **Baseline score** (released model, `turning_on_radio`, 27 full episodes) | **Q = 0.111** (3/27) | First real number; roughly matches the organizers' ~10% ([`baseline-Q-full.txt`](sessionA-2026-09-13/baseline-Q-full.txt)) |
| Episode length limit | **3,225 steps** | The charter assumed ~15,800, so every earlier cost estimate was ~5× too pessimistic |
| Scene loading | **~12 min warm, ~22 min cold, once per evaluator job** | Paid per job, not per episode, so evaluation is ~5× cheaper than feared ([`scene-reuse.jsonl`](sessionA-2026-09-13/scene-reuse.jsonl)) |
| Cost per extra instance in a job | **30.8 s** | Instances are cheap, jobs are not. An early "almost free" reading from n=3 was wrong ([`scale-n27.txt`](sessionA-2026-09-13/scale-n27.txt)) |
| End-to-end speed (simulator + policy inference) | **20.4 FPS** | Inside the organizers' 13.5–24.6 band |
| Evaluating k=8 tasks × n=27 instances | **≈ $19** | Evaluation cost does not constrain k |

Setup facts we paid to learn:
- **Rendering** only works on an EGL desktop image (`ghcr.io/selkies-project/selkies-egl-desktop:26.04`).
  The stock `runpod/pytorch` image cannot render ([`RENDERING-POSTMORTEM.md`](sessionA-2026-09-13/RENDERING-POSTMORTEM.md)).
- **Robot name:** a one-line compatibility fix is required for ANY run against this evaluator.
  Without it every run crashes with `KeyError: 'robot::proprio'`. It is now applied unconditionally
  (`training/patches/0001-robot-name-compatibility.patch`).
- **GPU memory is set opposite ways for the two jobs:**
  - **Serving next to Isaac Sim:** `XLA_PYTHON_CLIENT_PREALLOCATE=false`, fraction 0.35. JAX's default
    75% preallocation starves the simulator, and the error never mentions JAX.
  - **Training:** preallocate ON at 0.9.
- **Disk:** dataset download needs ~30 GB of free *container* disk for staging. The network volume
  (NFS) allocates ~2.5× apparent size.

## Data pipeline (CPU pods, 14 Sep)

Evidence: [`sessionB-2026-09-14-SLICE-VALIDATION.md`](sessionB-2026-09-14-SLICE-VALIDATION.md),
[`sessionB-2026-09-14-DATA-READY.md`](sessionB-2026-09-14-DATA-READY.md), [`DATASET_SLICING.md`](DATASET_SLICING.md).

Per-task slicing plus progress-label merge and compaction passed **all 10 checks on real data**.
Five bugs were caught that would have silently corrupted the experiment, with no error:
1. Every sample getting the first task's instruction (prompt table built from one dataset).
2. Progress labels landing on the wrong demonstrations. This spoils arm B only, so ΔQ would be meaningless.
3. Action chunks drawn from the wrong frames after slicing (frame ranges are absolute row positions).
4. Multi-task training silently dropping the progress column if any task lacked it.
5. The two arms training on different data. This is now impossible by mechanism (`--protocol`).

## Training (session B, RTX PRO 4500 Blackwell, 15–16 Sep)

Evidence: [`sessionB-2026-09-15-rung1/RUNG1.md`](sessionB-2026-09-15-rung1/RUNG1.md),
[`sessionB-2026-09-16-rung2-5/`](sessionB-2026-09-16-rung2-5/) (`README.md`, machine-generated
`RUNG2-5.md`, all training logs under `logs/`). Pass/fail rules were fixed in `AB_PROTOCOL.md`
revision 2026-09-15 **before** the runs.

- **The new GPU works.** JAX runs on Blackwell (sm_120).
- **Rung 1: both arms fit at batch 32, and the control really is the treatment minus the head.**
  - Arm A and arm B produced **the same action loss to 4 decimal places**.
  - Their normalisation statistics are **byte-identical**.
  - Decode-free norm stats are **bit-identical** to full decode on 640 seeded frames, and ~100× faster.
- **Memory:** arm B ran out of memory. Moving the per-term gradient measurement out of the update
  step cut peak use from 31.2 GB to 24.7 GB.
- **Rung 2:**
  - **The progress head learns:** loss 0.6971 → 0.6559, z = 18.7.
  - **It does not hurt the policy:** paired action-loss difference B − A = +0.0007, 95% CI [−0.0002, +0.0016].
- **Rung 3, λ:** at λ = 0.1 the progress term is **14–15% of the gradient**, so **λ ≈ 0.14–0.15
  gives 20%**. The two terms' gradients are almost orthogonal (cos ≈ +0.01).
- **Rung 4, speed (the expensive finding):**
  - median **3.90 s/step**, mean **8.11 s/step**, GPU idle at the median.
  - Training stalls **exactly every 24 steps, ~103 s each**, and those stalls are **52.8% of the
    wall clock**. It is a prefetch sawtooth: 8 workers × prefetch depth 3 = 24 batches.
  - The loader sustains 4.0 samples/s against the 8.2 a step needs.
  - Cost of 30k steps: **$48.68/arm as measured, $23.40/arm if the stall is removed** (~17+ workers predicted).
- **Rung 5:** two-task vs one-task speed ratio **1.00**, so k is a convergence and labelling question, not a cost one.

## What went wrong, and the lessons

- **The loader sweep was lost** ([`LOADER-SWEEP.md`](sessionB-2026-09-16-rung2-5/LOADER-SWEEP.md)).
  It wrote to container disk, the pod ran ~6 h unattended (~$4.30), then was deleted. The 24-worker
  fix is therefore **still untested**. Rule now: pod jobs write to the network volume and stop their own pod.
- **Two measurements nearly misled us**, both toward a comfortable answer:
  - Quoting only the median step time hid the stall.
  - The loader microbenchmark reported "777 items/s, KEEPS UP" while timing the prefetch queue
    draining. The training log is the trustworthy instrument.
- **The training logs were not on GitHub until 2026-09-18.** `*.log` was gitignored. Fixed, and
  `tests/test_evidence_is_tracked.py` fails if anything under `docs/` is ignored.
- **Two bugs in the shot-one runner** would have crashed arm B ~32 h into the run:
  - wrong config for arm B, and no label column;
  - a wrong flag for norm stats.
  Both were caught before any spend and are now pinned by tests that check the runner against the scripts it calls.

## Where things stand

**Ready to train:** [`scripts/shot_one.sh`](../scripts/shot_one.sh)
- Arms: coffee + shoes, 30k steps, λ 0.15, 24 workers.
- Checks both arms' complete configs before either trains.
- Stops at step 300 if the stall is still there ([`analysis/stall_gate.py`](../analysis/stall_gate.py)
  says NOGO on the real rung-4 log).
- Writes everything to the network volume and stops its own pod on every exit path.

The σ_w noise-floor runner (`scripts/session_a/noise_floor.sh`) has the same safeguards.

**Blocked on the RunPod balance** (HTTP 402 on 2026-09-18). The next round needs about **$50–55**:

| Step | Cost |
|---|---|
| Step-300 check | ~$0.75 |
| Both arms, if the check passes | ~$47 (hard cap $54) |
| Noise floor | ~$3 |

Nothing irreplaceable is on RunPod. The volume holds only the simulator environment and dataset,
which `scripts/setup_cloud.sh` and `scripts/download_dataset.sh` rebuild.
