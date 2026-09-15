# Notion "Findings" — update to publish

Staged here because the Notion connector was disconnected when this was written
(`/mcp` to reconnect). This is the text to append to the Findings page; it is kept
in the repo so the update is never lost with a session.

Page: https://app.notion.com/p/Findings-3d280ea5df6a81b49ce4db9e7fa38605

---

## Findings — 13–15 Sep 2026 (sessions A and B)

**Where we are:** the whole pipeline runs end to end on rented GPUs, the baseline is
measured, and both A/B arms train. Spend to date **$12.27 of $200**; deadline 16 Oct.
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
| Training speed (RTX PRO 4500, batch 32) | **~3.9 s/step best, ~10.4 s/step mean** | Sets the price of shot one |

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

### What is still open

- **Does the progress head actually learn?** Running now (1,000 steps per arm). The
  criterion was written down *before* the run: the last 100 steps must be below the
  first 100 by more than 2 standard errors.
- **λ (how much the progress term counts).** Will be calibrated from measured
  gradient shares, not guessed. The current 0.1 was set before anyone had seen a loss.
- **k (how many tasks).** Throughput cannot settle it; it is a convergence question.
  A practical limit: progress labels exist for **three** tasks, so k > 3 needs more
  labelling first.
- **Noise floor (σ_w).** Planned, not yet run; the protocol requires it before the
  first A/B.

### Cost discipline

Fourteen pods so far, every one terminated after use; the largest single spend was
$2.07. Cheap checks run before expensive ones — a $0.09 CPU box caught two data
defects that would otherwise have surfaced on a $0.72/hr GPU.
