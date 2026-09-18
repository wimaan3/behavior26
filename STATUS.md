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
| 10 | Shot one: train arm A and arm B | 🟡 **runner ready**: both arms proven before training, step-300 loader gate, self-stopping, results on the volume | `scripts/shot_one.sh` |
| 11 | Evaluate both arms, paired ΔQ (`analysis/compare.py`) | ⏳ blocked on 10 | — |
| 12 | Partial submission (2 tasks × 20 public instances) | ⏳ blocked on 10 | `submission/` |
| 13 | Write-up | ⏳ ongoing in `docs/` | — |

## Next actions, in order

1. **Shot one**, on one RTX PRO 4500 in EU-RO-1: `bash scripts/shot_one.sh` (coffee + shoes,
   30k steps, λ 0.15, 24 workers). It proves both arms' configs, then trains arm A. Its
   **step-300 gate** (`analysis/stall_gate.py`) measures whether 24 workers removed the
   rung-4 stall: GO continues (~$23/arm), NOGO stops the pod. Everything goes to the
   volume under `/workspace/shot1/`, and the pod stops itself on every exit path.
2. **σ_w noise floor** (§2, ~4 h ≈ $3): `bash scripts/session_a/noise_floor.sh` on an
   evaluator pod. Also writes to the volume and stops itself.
3. **Evaluate both arms** (paired ΔQ, `analysis/compare.py`), with `CONFIG=` set in
   `serve_baseline.sh`, sized by σ_w. Then submission and write-up.
4. The loader question from 2026-09-16 is answered by 1's gate. No separate sweep is needed.

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

Audited 2026-09-18. "Saved" means it survives this laptop and any pod.

| What | Where | Status |
|---|---|---|
| Code, scripts, tests, analysis | GitHub `wimaan3/behavior26` (public) | ✅ saved |
| Results, reports, protocol, this file | GitHub, `docs/` + `AB_PROTOCOL.md` | ✅ saved |
| Training/eval logs behind every result (23 files) | GitHub, `docs/session*/logs/` | ✅ saved since `c75ea72`. **Before that, `*.log` was gitignored and they existed only on the laptop.** A test now fails if anything under `docs/` is ignored |
| Norm stats (rung 1, both arms) | GitHub, `docs/sessionB-2026-09-15-rung1/norm_stats/` | ✅ saved |
| Progress labels (3 tasks) | GitHub, `labels/` | ✅ saved |
| Our openpi changes | GitHub, `training/patches/0001`, `0002` | ✅ saved. The local openpi clone was verified equal to upstream `0cc8e35` + these two patches, with nothing extra |
| Findings write-up for Notion | GitHub, `docs/NOTION-FINDINGS-UPDATE.md` | ⚠️ **not on Notion.** The connector returned 404 every time it was tried |
| Research notes: Discord sweep, handoff (6 files, ~120 KB, beside the repo) | **laptop only** | ❌ **not backed up.** They quote third-party Discord messages, so they need a PRIVATE home, not this repo |
| Simulator env + 36 GB dataset | RunPod network volume `96mu3d0s32` | rebuildable from `scripts/setup_cloud.sh` + `scripts/download_dataset.sh` |
| Pod container disks | nowhere | wiped on terminate. The 2026-09-16 loader sweep was lost this way (`docs/sessionB-2026-09-16-rung2-5/LOADER-SWEEP.md`) |

Rule going forward: a pod job that outlives the conversation writes its results to the
network volume and stops its own pod. Every session's logs are committed under `docs/`
before the pod is terminated.
