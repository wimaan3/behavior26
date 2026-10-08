# Shot two: the A/B on `turning_on_radio`, warm-started (fixed 8 Oct 2026, before any run)

## Why

Shot one's arms (10k steps from `pi05_base`, frozen VLM) never complete a task: 0 successes
on coffee and shoes, and in the videos the robot drives around without grasping
(base travel 82% of a human demo, hands 18%). The 25-pair shoes result (ΔQ +0.008,
p = 0.68) was measured at the floor. Shoes' partial credit is probably incidental (not
confirmed), so it is a poor measuring stick.

`turning_on_radio` is the one task where a policy succeeds through our serving and
evaluation stack: the released checkpoint scored Q = 0.111 over 27 instances. It is
all-or-nothing (D = 1), so there is no accidental partial credit.

## Design

| | Arm A (control) | Arm B (treatment) |
|---|---|---|
| Config | `pi05_b1k_frozen_vlm` | `pi05_b1k_frozen_vlm_progress`, λ 0.15 |
| Initial weights | released radio checkpoint | same (progress head initialised fresh) |
| Norm stats | the released checkpoint's own file | same file |
| Data | radio, 200 demos, 429,928 frames | same, plus `progress` labels |
| Steps, batch, seed | 6,000 × 32, seed 42 | same |
| Prompt, train and serve | `turning_on_radio` (the task name) | same |

The progress label on a D = 1 task is binary: 0 until the radio is on, 1 after. 27.9% of
frames are 1 (demos continue ~600 frames after the switch), so the head learns "is the task
done yet". This is a weaker signal than on a graded task. It is what this task offers.

One runner (`scripts/shot_one.sh` with `INIT_FROM=released STATS_FROM=released
TASKS_CSV=turning_on_radio`) trains both arms through one trainer call.

## Checks that stop the run (in the pod)

- Warm-start rule (`analysis/health_gate.py --warm-start`): action loss over the first 20
  steps must be below 0.30 (shot one from `pi05_base` opened at 0.94), and still below 0.30
  at step 1,000. Catches a checkpoint that did not load or mismatched norm stats ~2 minutes in.
- Arm B: Branch 0 on the progress loss at step 1,000, unchanged.
- Loader speed gate, resume test, NaN check, hours cap: unchanged.

## Evaluation and analysis (fixed now)

- `configs/experiments/002-radio-ab.yaml`: 27 training instances (10–36) × 4 attempts, both
  arms, 108 attempts per arm.
- Primary result: per-instance mean Q (= success rate), paired difference B − A over the 27
  instances, two-sided 95% CI and p-value from `analysis/compare.py`. Significant means
  p < 0.05.
- The first 9 instances (10–18) run first for both arms. The only decision taken on that
  look: if both arms have zero successes in their 36 attempts, stop and report, because the
  fine-tune broke the policy. No stopping for a good-looking difference.
- One evaluation. It is not repeated or extended until it is significant.
- Reference, descriptive only, if budget remains: the released checkpoint, unchanged, with
  the same prompt, one attempt per instance.

With 108 attempts per arm, a change from about 11% to about 25% success is detectable; a
smaller effect will read as no difference, and will be reported as that.

## Budget

Balance at the start: $37.20. Training ≈ 14 h at $0.72/h ≈ $10; evaluation ≈ $12–14;
volume ≈ $0.47/day. No funds are added by the agent.
