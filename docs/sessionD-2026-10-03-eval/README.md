# Session D: evaluating shot one, 3–4 Oct 2026

Goal: paired ΔQ between arm A (control) and arm B (progress head) on the frozen dev
subset (coffee + shoes, training instances 10–36), with video of every attempt.

**Outcome: the pipeline works end to end, but there is no usable A/B result yet.** The
budget ran out on the wrong scope: see "What went wrong".

## Results

| Arm | Task | Attempts | Successes | Mean Q | Attempts with Q > 0 |
|---|---|---|---|---|---|
| A | coffee | 13 | 0 | **0.000** | 0 |
| B | coffee | 3 | 0 | **0.000** | 0 |
| A | shoes | 12 | 0 | **0.042** | **4** (Q = 0.1, 0.1, 0.2, 0.1) |
| B | shoes | 0 | — | — | — |

Per-attempt JSON: [`results/`](results/). Logs: [`logs/`](logs/). Sample video:
[`video/armA_shoes_15_q0.2.mp4`](video/armA_shoes_15_q0.2.mp4), the best attempt. All 31
videos are on the owner's laptop at `~/behavior26-results/eval1/` (187 MB).

- **Coffee is at the floor** for both arms at 10k steps: no attempt satisfied a single
  goal predicate. The task cannot show a difference between the arms.
- **Shoes has signal.** Arm A gets partial credit on a third of instances (one or two
  shoes placed). **Arm B on these same 12 instances is the comparison that matters**, and
  it has not been run.
- The 3 coffee pairs (instances 10–12) are 0 vs 0. That's ΔQ = 0, and it is uninformative.

## The pipeline, proven

The first completed attempt (arm A, coffee, instance 10) proved the loop: Isaac Sim →
policy server serving our checkpoint → scored JSON → video. Before that, four defects were
found and fixed, each caught early:

1. **The smoke test fed a stale sample observation.** openpi's `make_b1k_example` has a
   23-number state, while `B1KInputs` indexes the full proprio vector (53+). It now builds the input
   from the policy's own robot config. Cost: none; caught before Isaac Sim.
2. **Our tasks were not in openpi's prompt registry**, so every server died on `KeyError`.
   Patch 0003 registers them **with the exact training prompts, which are the task
   names**. Training used `prompt_from_task=True`, and `tasks_from_metadata` maps
   task_index to the name in `meta/tasks.parquet`. The released baseline appears to have
   the opposite mismatch: trained on `turning_on_radio`, served a full sentence. The
   runner now checks prompts before Isaac Sim.
3. **The evaluator launch discarded `--task-name`.** It was a `bash -c` + `shift` bug. An
   executed test runs the real launch line against a stub.
4. **Isaac Sim segfaulted right after "app ready"** on a pod using
   `selkies-egl-desktop:26.04`. That is a **moving tag**: it pointed at a 2 Oct rebuild.
   **Pinning `26.04-20260912122125`**, the build current when session A's evaluation
   worked, fixed it. **Always pin this tag.**

## What went wrong

- **Simulation is ~5× slower than assumed.** Coffee ran at **3.96 FPS**: 9,400 steps in
  39.5 min. The 20.4 FPS used for budgeting was measured on `turning_on_radio`, a small
  scene. Same class of error as the episode limit: one task's number used for all tasks.
  The full protocol (27 instances × 2 arms × 2 tasks) would be ~80 h ≈ $59, not ~$14.
- **The simulator crashed mid-unit twice and exited 0.** The errors were OmniGibson
  `get_joint_positions` → `'NoneType' object has no attribute 'view'`, and "Exception
  occurred during pre-physics step". The runner recorded those units as INCOMPLETE and
  moved on, which is how it ended up on shoes arm A instead of finishing coffee pairs.
- **The run kept going while a scoping question waited 20 h for an answer.** It spent
  the budget in the original order (all of arm A first) and hit its 21 h cap. Lesson: pause
  or stop an expensive run before any blocking question.

## Cost

Evaluation pods 3–4 Oct: **$17.78** of the $20 funded (3 short failed attempts, then
one 21 h run), plus ~$0.10 to pull the results. **Total project spend ≈ $72.**

## Next, if funded

**Arm B on shoes, instances 10–21**: the same 12 instances arm A already has. That is
12 × ~49 min + scene load ≈ 10.5 h ≈ **$8**. It gives a real paired comparison on the
task where the policy shows partial progress. The runner needs two changes first:
- an instance subset and task override;
- re-invoking the evaluator for the missing instances when the simulator crashes
  mid-unit, instead of moving on.
