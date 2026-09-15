# behavior26 — project status

**Read this first.** One page tracking every deliverable. Detailed evidence lives in
the dated `docs/session*` folders it links to. Updated with every milestone.

*Last updated: 2026-09-15 · Deadline: **16 Oct 2026** (31 days) · Spend: **$12.27 of $200***

## Deliverables

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Rendering + eval stack works on rented GPU | ✅ done | `docs/sessionA-2026-09-13/RENDERING-POSTMORTEM.md` |
| 2 | Evaluation cost model (scene reuse, per-instance, fps) | ✅ done | `AB_PROTOCOL.md` revs 13e–13g, 2026-09-14 |
| 3 | Baseline Q on the released checkpoint | ✅ **0.111** (3/27) | `docs/sessionA-2026-09-13/baseline-Q-full.txt` |
| 4 | Per-task dataset slicing + label merge + compaction | ✅ done, validated on real data | `docs/DATASET_SLICING.md`, `docs/sessionB-2026-09-14-DATA-READY.md` |
| 5 | Progress labels for training tasks | ✅ 3 tasks (coffee, shoes, toolbox) on `main` | `labels/` |
| 6 | Session B rung 1 — both arms load, fit, 10 steps | ✅ done, batch 32 | `docs/sessionB-2026-09-15-rung1/RUNG1.md` |
| 7 | Session B rungs 2–5 — head learns, λ, steps/s, k | 🟡 **script ready, not yet run** | `scripts/session_b/rung2_5.sh`, pre-registered in `AB_PROTOCOL.md` rev 2026-09-15 |
| 8 | Noise floor σ_w (§2: 12 instances × 6 repeats) | 🔴 **not started** — protocol says before the first A/B | `AB_PROTOCOL.md` §2 |
| 9 | Decide k and λ | ⏳ blocked on 7 (and on labels if k > 3) | — |
| 10 | Shot one: train arm A and arm B | ⏳ blocked on 7, 8, 9 | — |
| 11 | Evaluate both arms, paired ΔQ (`analysis/compare.py`) | ⏳ blocked on 10 | — |
| 12 | Partial submission (2 tasks × 20 public instances) | ⏳ blocked on 10 | `submission/` |
| 13 | Write-up | ⏳ ongoing in `docs/` | — |

## Next actions, in order

1. **Run rungs 2–5** on one RTX PRO 4500 pod (~3 h, ~$2.20): `bash scripts/session_b/rung2_5.sh`,
   then `python -m analysis.rung_report <log dir>` and commit the report.
2. **Run the σ_w noise floor** (§2) — needed to size the A/B; cheap now that scene load is per job.
3. **Decide k and λ** from the rung report. k > 3 needs more Jetson labels — start those
   in parallel if k > 2 is still wanted.
4. **Shot one**, then evaluation, submission, write-up.

## Open decisions for the project owner

- **k:** throughput cannot settle it (batch-bound by construction); it is a convergence
  and labelling question.
- **Research notes outside the repo** (`../HANDOFF-START-HERE.md`, Discord sweep notes)
  are not on GitHub. They quote third-party Discord messages and this repo is public,
  so they have not been pushed without an explicit decision.

## Where results are saved

Every pod session's logs, fingerprints and artifacts are salvaged before the pod is
terminated and committed under `docs/session*`. Nothing irreplaceable lives on a pod
or on the network volume.
