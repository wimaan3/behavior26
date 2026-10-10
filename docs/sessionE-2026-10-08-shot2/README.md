# Session E: shot two, the radio A/B, 8–10 Oct 2026

Plan and analysis were fixed before any run: [`PLAN.md`](PLAN.md).

## Result

Both arms were initialised from the released `turning_on_radio` checkpoint and fine-tuned
for 6,000 steps on radio; arm B adds the progress head (λ 0.15). Evaluation: 27 training
instances (10–36) × 4 attempts per arm, 108 attempts each, same GPU type for both arms.

| | Successes | Success rate (mean Q) |
|---|---|---|
| Arm A (control) | 38 / 108 | 0.352 |
| Arm B (progress head) | 40 / 108 | 0.370 |
| **ΔQ (B − A), paired over 27 instances** | | **+0.019**, 95% CI [−0.091, +0.128], p = 0.73 |

**Not significant.** The progress head makes no measurable difference on this task. The
design could detect |ΔQ| ≥ 0.11. Full table: [`compare.txt`](compare.txt); per-attempt
JSON: [`results/`](results/).

**Our own trained models complete the task.** Both arms succeed on about a third of
attempts. Outcomes are driven by the instance: 5 instances succeed almost every time for
both arms, 7 never succeed for either.

For reference only, not a controlled comparison: the released checkpoint scored 3 / 27
(Q = 0.111) on 13 Sep, on different instances (0–26), one attempt each, a different GPU and
upstream's sentence prompt. A two-attempt check of the released checkpoint with our prompt
on instance 10 (`results/ref_released/`) failed both times; instance 10 also failed all
four of arm A's attempts.

## Why the head did not matter here

`turning_on_radio` has one goal condition, so the progress label is binary (0 until the
radio is on, 1 after; 27.9% of frames are 1). The head learned it almost at once: progress
loss 0.578 → 0.023 by step 1,000 and ~0.001 afterwards. With its loss near zero it sends
almost no gradient into the policy for the remaining 5,000 steps, so the two arms train
nearly identically. The task that lets the policy succeed is the task that gives a progress
head the least to say. A graded task would test the head properly, but shot one showed our
budget cannot train a policy that succeeds on one.

## Training

One runner, one trainer call, both arms ([`logs/manifest.json`](logs/manifest.json)):
RTX PRO 4500, 3.71 s/step, batch 32, seed 42, norm stats reused from the released checkpoint.

| | Opening action loss | At step 1,000 | Final (step 5,999) |
|---|---|---|---|
| Arm A | 0.080 | 0.031 | 0.015 |
| Arm B | 0.080 | 0.032 | 0.022 |

Shot one, from `pi05_base`, opened at 0.94. Resume test, loader gate and both health checks
passed. Final-step losses are single noisy readings, not a difference between arms.

## What went wrong, and what it cost

- Three launches failed in minutes before training started: volume space (grown 200 → 250
  GB), and two bugs that only appear with a single task (the merge wrote no filter manifest
  when nothing was dropped; the trainer was given the parent directory as dataset root).
  Each is fixed with a test.
- The first evaluation block hit a 7-hour cap that was set too low; the pod was terminated
  and the evaluation resumed on a new pod of the same type. No result was lost. One
  instance's arm B attempts (instance 14) were rerun from scratch by the resume rule.
- **Videos are incomplete.** With the evaluator still assuming a 200 GB volume, the video
  budget read as spent, so arm A's first 36 videos (instances 10–18) stayed on the pod and
  were lost; arm B's first 16 (instances 10–13) were lost the same way. 164 videos survive
  (72 arm A, 92 arm B), on the owner's laptop in `~/behavior26-results/shot2/videos`,
  sha256-verified. They are not in this repository.
- Radio evaluation took ~7.5 min per attempt on this GPU, about twice the estimate.

Spend for shot two: about $31.70 of the $37.20 balance (training ≈ $10.50, evaluation
≈ $20, a GPU check and transfers ≈ $1). Balance afterwards: $5.52. No pods running.

## What this does and does not support

- Supports: the full pipeline (data, warm-start training, serving, evaluation, paired
  analysis) works end to end, and a policy we trained completes a BEHAVIOR task.
- Does not support: any claim that the progress head helps. Two A/Bs (shoes at the floor,
  radio at ~36%) both read as no difference.
- Per PLAN.md the evaluation is not repeated or extended to look for significance.
