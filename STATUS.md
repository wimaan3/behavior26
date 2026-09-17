# behavior26 — project status

**Read this first.** One page tracking every deliverable. Detailed evidence lives in
the dated `docs/session*` folders it links to. Updated with every milestone.

*Last updated: 2026-09-16 · Deadline: **16 Oct 2026** (30 days) · Spend: **$23.61 of $200** · Pods running: **none***

## Deliverables

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Rendering + eval stack works on rented GPU | ✅ done | `docs/sessionA-2026-09-13/RENDERING-POSTMORTEM.md` |
| 2 | Evaluation cost model (scene reuse, per-instance, fps) | ✅ done | `AB_PROTOCOL.md` revs 13e–13g, 2026-09-14 |
| 3 | Baseline Q on the released checkpoint | ✅ **0.111** (3/27) | `docs/sessionA-2026-09-13/baseline-Q-full.txt` |
| 4 | Per-task dataset slicing + label merge + compaction | ✅ done, validated on real data | `docs/DATASET_SLICING.md`, `docs/sessionB-2026-09-14-DATA-READY.md` |
| 5 | Progress labels for training tasks | ✅ 3 tasks (coffee, shoes, toolbox) on `main` | `labels/` |
| 6 | Session B rung 1 — both arms load, fit, 10 steps | ✅ done, batch 32 | `docs/sessionB-2026-09-15-rung1/RUNG1.md` |
| 7 | Session B rungs 2–5 — head learns, λ, steps/s, k | ✅ **done** — Branch 0 LEARNING, action loss NOT_DEGRADED | `docs/sessionB-2026-09-16-rung2-5/` |
| 8 | Noise floor σ_w (§2: 12 instances × 6 repeats) | 🔴 **not started** — protocol says before the first A/B | `AB_PROTOCOL.md` §2 |
| 9 | Decide k and λ | 🟡 λ ≈ **0.14–0.15** for a 20% gradient share; k still open (labels, not throughput) | `docs/sessionB-2026-09-16-rung2-5/RUNG2-5.md` |
| 10 | Shot one: train arm A and arm B | ⏳ blocked on 7, 8, 9 | — |
| 11 | Evaluate both arms, paired ΔQ (`analysis/compare.py`) | ⏳ blocked on 10 | — |
| 12 | Partial submission (2 tasks × 20 public instances) | ⏳ blocked on 10 | `submission/` |
| 13 | Write-up | ⏳ ongoing in `docs/` | — |

## Next actions, in order

1. **Settle the loader. NOT answered: the sweep was lost with the pod.** Rung 4 spends 52.8%
   of the clock in a prefetch sawtooth, and the arithmetic predicts ~17+ workers removes it
   (30k steps: $48.68 → $23.40 per arm). The corrected sweep ran on container disk. The pod
   was then terminated before anyone copied the log, and billed ~6 h unattended first. See
   `docs/sessionB-2026-09-16-rung2-5/LOADER-SWEEP.md` for the timeline and a re-run
   recipe that writes to the volume and stops its own pod.
2. **Run the σ_w noise floor** (§2, ~4 h ≈ $3) — needed to size the A/B. Built and tested
   (`scripts/session_a/noise_floor.sh`, `analysis/noise_floor.py`), not yet run.
3. **Decide the step budget** once 1 lands, then **k**. λ is answered (≈0.14–0.15 for a
   20% gradient share); k is a labelling and convergence question, not a throughput one.
4. **Shot one**, then evaluation, submission, write-up. The runner is built and tested
   (`scripts/shot_one.sh`: one shared trainer call, resumable, seed recorded and applied),
   and `serve_baseline.sh` can now serve our own arms (`CONFIG=pi05_b1k_frozen_vlm`).

## Open decisions for the project owner

- **Step budget for shot one.** At the measured mean (8 workers) 30k steps is $48.68
  per arm and 10k is $16.23; if the sawtooth fix lands they are $23.40 and $7.80. Two
  arms plus the σ_w floor plus evaluation has to fit in the $182 left. Recommendation:
  decide after the loader sweep, and prefer 30k-if-fixed over 10k-anyway — a
  null result from an undertrained arm answers nothing.
- **k:** throughput cannot settle it (ratio 1.00, batch-bound by construction); it is a
  convergence and labelling question. Three tasks are labelled today.
- **Research notes outside the repo** (`../HANDOFF-START-HERE.md`, Discord sweep notes)
  are not on GitHub. They quote third-party Discord messages and this repo is public,
  so they have not been pushed without an explicit decision.

## Where results are saved

Every pod session's logs, fingerprints and artifacts are salvaged before the pod is
terminated and committed under `docs/session*`. Nothing irreplaceable lives on a pod
or on the network volume.
