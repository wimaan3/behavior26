# Notion "Findings" — update to publish

Staged here because the Notion connector was disconnected when this was written
(`/mcp` to reconnect). This is the text to append to the Findings page; it is kept
in the repo so the update is never lost with a session.

Page: https://app.notion.com/p/Findings-3d280ea5df6a81b49ce4db9e7fa38605

---

## Findings — 13–16 Sep 2026 (sessions A and B)

**Where we are:** the whole pipeline runs end to end on rented GPUs, the baseline is
measured, and both A/B arms train. Spend to date **$18.00 of $200**; deadline 16 Oct.
Everything below is reproducible from the repo — evidence links are files, not claims.

### The headline numbers

| Finding | Value | Why it matters |
|---|---|---|
| Baseline Q, `turning_on_radio`, n=27, full episodes | **0.111** (3/27 successes) | First real number; corroborates the organizers' ~10% figure |
| Episode timeout | **3,225 steps** | The charter assumed ~15,800 — 5× too high, so every cost estimate built on it was pessimistic |
| Scene load | **~12 min warm, ~22 min cold**, paid **once per job** | Not per rollout. Evaluation cost fell ~5× |
| Per-instance cost after reuse | **30.8 s** | Instances are cheap; jobs are not |
| End-to-end eval throughput | **20.4 FPS** | Inside the organizers' 13.5–24.6 band |
| Evaluation for k=8, n=27 | **≈ $19** | k is no longer gated by evaluation cost |
| Training speed (RTX PRO 4500, batch 32) | **3.90 s/step median, 8.11 s/step mean** | Sets the price of shot one; the gap is the finding, see below |
| Does the progress head learn? | **Yes** — loss 0.6971 → 0.6559, z = 18.7 | The A/B is measuring something real |
| Does the head damage the policy? | **No** — paired B−A = +0.0007, CI [−0.0002, +0.0016] | The treatment is additive, not a trade |
| λ for a 20% gradient share | **≈ 0.14–0.15** (0.1 gives ≈14–15%) | Calibrated from measured gradients, not guessed |

### The five things that would have silently corrupted the experiment

Each was caught before it could reach a result, and each is now pinned by a test.

1. **Prompt lookup by task index.** openpi builds its prompt table from the *first*
   dataset only. Renumbering task ids per slice — the obvious design — would have
   given every sample the first task's instruction, with no error.
2. **Progress labels landing on the wrong demonstrations.** LeRobot v3 requires
   densely numbered episodes, but the Jetson labels are keyed to the original
   numbering. Arm A would have been fine and arm B silently poisoned — ΔQ meaningless.
   Fixed by carrying the original index as a column that travels with the rows.
3. **Action chunks drawn from the wrong frames** after slicing, because episode frame
   ranges are absolute row positions.
4. **Multi-task training silently dropping the progress column** if any one task
   lacked it — arm B would train with no target at all.
5. **Arm A and arm B on different data.** Now a mechanism, not a rule: a declared arm
   refuses to start unless it points at the merged, filtered root.

### Evidence that the arms are comparable

At rung 1, arm A and arm B produced **identical action-loss values to four decimals**
at every step, diverging only in the fourth decimal after step 4 — same seed, same
batches, same initialisation. The control really is the treatment minus the head.
Both arms' normalisation statistics are **byte-identical** (sha256 `9a3dcf3f…`).

### The rung-4 finding: half the run is one repeating stall

Arm A trains at a **median of 3.90 s/step** but a **mean of 8.11 s/step**, with the GPU
at **0% utilisation**. Quoting the median would have underpriced shot one by ~40%.

The step numbers say what to fix. 39 intervals exceeded 20 s, and they land on steps
72, 96, 120, … — **exactly every 24 steps**, ~103 s each, together **52.8% of the entire
wall clock**. 24 is the dataloader pipeline depth (8 workers × prefetch 3), not anything
about the data: the trainer drains a full queue at the GPU-bound rate, then blocks while
the workers refill it. Sustained loader rate is **4.0 samples/s** against the **8.2** a
3.90 s step consumes.

So this is a worker-count problem, and it is worth real money:

| | 10k steps | 30k steps |
|---|---|---|
| at the measured mean (8 workers) | $16.23/arm | $48.68/arm |
| if the sawtooth is removed | $7.80/arm | $23.40/arm |

The measurement nearly went wrong twice, both times in the direction of a *comfortable*
answer. The report first quoted only the median, which is blind to a stall landing on 4%
of steps. The loader microbenchmark first reported **777 items/s, "KEEPS UP"** — it was
timing the prefetch queue draining at memory speed rather than the loader producing.
Both are now fixed and pinned by tests, and the rule learned is that **the training log
is the instrument**: a 1000-step run contains 39 independent refill cycles, while a
70-batch microbenchmark cannot outrun a 72-deep queue.

### What is still open

- **The loader fix.** A corrected sweep (8/16/24 workers × pyav/torchcodec) is running
  now. It sets the step budget for shot one.
- **k (how many tasks).** Throughput cannot settle it — one-task and two-task steps/s
  came out at a ratio of **1.00**. It is a convergence question, and a labelling one:
  progress labels exist for **three** tasks, so k > 3 needs more labelling first.
- **Noise floor (σ_w).** Built and tested, not yet run; the protocol requires it before
  the first A/B.
- **Step budget.** The live decision. A null result from an undertrained arm answers
  nothing, so the preference is 30k-if-the-loader-is-fixed over 10k-anyway.

### Cost discipline

Twenty-one pods so far, every one terminated after use; the largest single spend was
$5.86 (the session-B training box). Cheap checks run before expensive ones — a $0.09 CPU box caught two data
defects that would otherwise have surfaced on a $0.72/hr GPU.
