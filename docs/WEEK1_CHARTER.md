# BEHAVIOR Challenge 2026 — Week 1 Charter

> **CORRECTION, 2026-09-04.** This charter was written before we had the
> evaluator source. Three of its numbers are wrong and are struck through below,
> with the v3.9.2 values inline. Source of truth:
> `external/b1k/OmniGibson/omnigibson/eval/utils/eval_utils.py`.
>
> - Test instance ids are **301–340** (public 301–320, hidden 321–340), not 0–19.
>   Everything below 301 is a *training* instance.
> - A full public submission is **2,000** rollouts (100 tasks x 20 instances),
>   not 1,000. `compute_final_q_score` divides by 20 per task regardless of how
>   many were run, so a partial submission is scored with zeros, not omissions.
> - That is **~700–840 GPU-hours**: ~7–9 days on 4 cards, ~3.5–4.5 days on 8.

**Deadline: 16 October 2026. Today: 26 August 2026. Time remaining: 7 weeks, 2 days.**

This document covers three things: how to review the existing plan, what Week 1 must produce, and how to set up the environment and tooling.

---

## 0. Read this first: the timeline correction

The original plan is drawn against a 14-week calendar starting October 2025. The real challenge launched 2 July 2026 and closes 16 October 2026. **You have half the time the plan assumes.**

Phases 1–3 of the original plan (Setup, Analysis, Data & Prototyping) budget five weeks before model development begins. That has to become about one week. Everything below is built around that compression.

---

## 1. How to review the plan

Do this as one focused 90-minute session, not a series of messages. Bring the poster up on screen and work through it in order.

### Session structure

| Time | Activity |
|---|---|
| 10 min | Agree on what success means for you two (§2 below) |
| 20 min | Walk the poster box by box, mark each **Keep / Revise / Cut** |
| 30 min | Close decisions D1–D5 (§3) |
| 20 min | Assign Week 1 owners and acceptance criteria (§4) |
| 10 min | Write every decision into the Decision Log — with the date and the reasoning |

The rule for the session: **no decision gets deferred without a named owner and a close-by date.** Deferred decisions are what kill 7-week projects.

### Suggested verdicts going in

**Keep — these are good and shouldn't be relitigated:**

- The core thesis (predicate-aware VLA). It's well-aimed, and last year's winner independently built the same kind of stage-tracking component on top of π0.5. That's strong validation.
- The system architecture diagram in box 1. Accurate, including the important detail that BDDL task definitions *are* available at evaluation time.
- The evaluation strategy in box 5 (protect against overfitting by keeping a frozen dev set). The *instinct* is correct and disciplined; the ~~instance numbers (report on 0–9, dev on 10–19)~~ are wrong — those are training instances, and there is no holdout inside the public test set. Dev loop runs on training instances; 301–320 are all scored.
- The risk table in box 7. The four risks listed are real.
- The workflow loop in box 3, especially the "optional small curated failure-recovery dataset" note — that's a genuinely sharp observation, because all 20,000 demos are successes and the model has never seen a recovery.

**Revise:**

- **The 6-phase timeline.** Halve it. See §5.
- **Scope of the four contribution components.** Four is too many for 7 weeks. See D5.
- **The resources box.** "We train on more powerful hardware" needs to become a provider, a budget, and a card.

**Cut:**

- **"Submit interest to the program."** There is no registration step for this challenge. Nothing to submit interest to. Replace with: join the Discord and attend Monday office hours (5–6pm Pacific).
- **"Final evaluation (1,000 rollouts per task)."** Wrong by 100×. It is ~~1,000~~ **2,000** rollouts *total* — 100 tasks × ~~10~~ **20** instances × 1 rollout each. Correct it so nobody plans compute against the wrong number. (The 10-instance figure here was itself wrong; see the correction banner.)

---

## 2. Redefine success before anything else

The plan's success criterion is "Top 5 finalist." Here is last year's actual leaderboard, held-out Q-score:

| Rank | Team | Q | Success rate |
|---|---|---|---|
| 1 | Robot Learning Collective (independent) | 0.260 | 12.4% |
| 2 | Comet (NVIDIA Research) | 0.251 | 11.4% |
| 3 | SimpleAI Robot | 0.159 | 10.8% |
| 4 | The North Star (Huawei) | 0.120 | 7.6% |
| 5 | Embodied Intelligence | 0.095 | 5.2% |
| 9–18 | ten other teams | 0.014 → 0.000 | ~0 |

Four teams scored **exactly zero**. The winner completed 12% of tasks. This is a brutally hard benchmark with a very top-heavy distribution.

Set three tiers instead of one:

- **Floor — a valid, verified submission on the leaderboard with non-zero Q.** Sounds modest. Four teams failed to clear it last year, and others never submitted at all. Protect this above everything.
- **Target — mean Q ≥ 0.10.** That would have placed 4th–5th last year. Achievable for two people with cloud compute and good execution.
- **Stretch — the $1,000 Outstanding Open Source prize.** This is the one genuinely under-contested prize in the pool. Of last year's top five, only two released code. Nobody optimizes for this. You can.

**Be honest about first place.** You are two people with one 3090 and seven weeks, against NVIDIA Research and teams with sponsored multi-GPU compute. You cannot win the scaling race. You can win on execution, reproducibility, and public usefulness.

### The scoring insight that should shape your strategy

Your ranking metric is **mean Q across all 100 tasks**, and Q is *partial credit over goal predicates*.

That means a policy that reliably completes the **first one or two predicates of all 100 tasks** scores far better than one that perfectly solves 15 tasks and zeroes the other 85. Do the arithmetic: 20 tasks at Q = 0.5 with 80 zeros gives mean Q = 0.10. Every task at Q = 0.15 gives mean Q = 0.15 — with far less depth required.

**Broad shallow competence beats narrow depth here.** This argues for multi-task training across all 100 tasks, and it means the predicate/progress idea is pointed at the right target.

---

## 3. Decisions to close in Week 1

Each of these blocks downstream work. Each needs an owner and a close-by date.

| # | Decision | Recommendation | Close by |
|---|---|---|---|
| **D1** | Base model: π0.5 or GR00T N1.7? | **π0.5.** 1st and 2nd place both used it; GR00T did not place. The `wensi-ai/openpi` fork is maintained by a challenge organizer. GR00T additionally requires a gated Cosmos-Reason2 backbone and an 8-GPU reference recipe. | Day 2 |
| **D2** | Cloud provider and hard budget ceiling | On-demand 4090/A100 with a **persistent network volume** (you cannot re-download 3 TB per instance). RunPod or Lambda are the pragmatic picks. Set a ceiling in dollars, today. See §6 for estimates. | Day 3 |
| **D3** | Data scope and storage | Start with a **10-task slice (~330 GB)** to build the pipeline end to end. Decide on the full 3.27 TB download only after measuring your actual throughput. | Day 3 |
| **D4** | Task scope: subset or all 100? | **All 100, multi-task.** Justified by the partial-credit arithmetic above, and by last year's finding that multi-task training produces *emergent recovery* that single-task models never show. | Day 5 |
| **D5** | Contribution scope — which of the four components? | **Components 1–2 (predicate head + progress head) are the committed contribution.** Component 3 (failure detection) if time allows. Component 4 (recovery conditioning) is the stretch, because it's the only one that changes runtime behavior and carries real engineering risk. | Day 7 |
| **D6** | Success definition | Adopt the three tiers in §2. | Day 1 |

### One unexplored option worth 10 minutes of discussion

**The 2026 challenge does not fix the robot embodiment.** You may pass `--robot-config` with any OmniGibson-supported robot; R1Pro is merely the default. This is new this year and nobody has explored it publicly.

Almost certainly you should stay on R1Pro — all 20,000 demos are R1Pro, so switching throws away your entire training set. But the *controller configuration* within R1Pro is also yours to change (joint vs. IK vs. base controllers), and that's a cheap knob most teams won't touch. Note it, don't chase it.

### Take the free wins before the clever ones

Before any novel work, bank what last year's winner already proved out:

1. **Gripper auto-reopen heuristic.** If the gripper fully closes on nothing, reopen it. The demo data contains only successful trajectories, so the model never learned to retry a failed grasp. Zero training cost, meaningful gain.
2. **Multi-task training over all 100 tasks** (see D4).
3. **Correlated noise for flow matching**, if it ports cleanly into the fork.

These are cheap, proven, and independent of your contribution. Land them first so you always have a working submission to fall back on.

---

## 4. Week 1 deliverables

Both of you install the stack (days 1–2) — you both need to understand it. Then split by strength.

**Proposed ownership split:** you take infrastructure, measurement and analysis; your teammate takes model, data and training. This plays to who wrote the plan and who's buying the GPU, and it puts the highest-leverage unglamorous work (the harness) in dedicated hands.

| ID | Deliverable | Owner | Done when |
|---|---|---|---|
| **W1-A** | One cloud GPU box running BEHAVIOR-1K at the current pinned tag | Both | `conda activate behavior` works and a scene loads headless |
| **W1-B** | One full rollout of `turning_on_radio` using the released π0.5 checkpoint | Both | A `q_score` JSON and an MP4 exist on disk |
| **W1-C** | **The number: measured wall-clock per rollout** | You | A timing figure with scene-load and step time broken out separately |
| **W1-D** | Repo scaffold + W&B project + Docker skeleton | You | `docker build` succeeds and the container serves `/healthz` |
| **W1-E** | Data audit — 10-task slice downloaded, size and read throughput measured | Teammate | Numbers written into the Decision Log |
| **W1-F** | Decision Log with D1–D6 closed | Both | Six entries, each with date and reasoning |

**W1-C is the single most important output of the week.** Everything downstream — how many experiments you can run, whether a full submission is even feasible, how much cloud budget you need — falls out of that one measurement. Do not let the week end without it.

### Why the harness matters more than it looks

Rough estimate, to be replaced by your W1-C measurement:

- Full-res RGB+depth evaluation runs at **~13.5 FPS** on a 4090 (organizers' published benchmark)
- Scene load is **150–300 seconds per rollout**
- Default timeout is **1.5× mean human demo length**; mean demo is 351 s
- Assuming 30 Hz control: ~15,800 steps → **roughly 20–25 minutes per rollout**

**2,000 rollouts ≈ 700–840 GPU-hours ≈ ~7–9 days on 4 cards, ~3.5–4.5 days on 8.**

One full leaderboard submission is a multi-week job on one card. You need parallel eval workers or you will not submit at all. This is why the harness is Week 1 work and not Week 5 work — and it's also why it's a credible candidate for your standout open-source contribution. It's the piece everyone needs and nobody publishes.

---

## 5. Reshaped timeline

| Week | Dates | Focus |
|---|---|---|
| **1** | Aug 26 – Sep 1 | Setup, one rollout end to end, **measure**, close D1–D6, repo + Docker skeleton |
| **2** | Sep 2 – 8 | Parallel eval harness working. Baseline π0.5 fine-tune launched on a task subset. Free-win heuristics implemented. |
| **3** | Sep 9 – 15 | **First real submission attempt**, however weak. Failure taxonomy from real rollout data. Predicate/progress label extraction. |
| **4** | Sep 16 – 22 | Predicate + progress heads trained. First ablation: with vs. without. |
| **5** | Sep 23 – 29 | Full 100-task multi-task training run. Recovery conditioning if D5 allows. |
| **6** | Sep 30 – Oct 6 | Scale up, tune, robustness. Second full submission. |
| **7** | Oct 7 – 13 | Final model selection. Full 1,000-rollout evaluation. Package Docker + JSONs + videos. |
| **Buffer** | Oct 14 – 16 | Submit. Do not plan work here. |

**Non-negotiable:** get a valid submission on the leaderboard in Week 3, even if the score is terrible. A working weak submission that improves is infinitely better than a great model that misses packaging. The submission pipeline is itself a thing that breaks, and you want to discover how in Week 3, not on 15 October.

---

## 6. Environment and tooling

### Hardware reality

| Asset | Verdict |
|---|---|
| Your laptops (no GPU) | Code, orchestration, analysis only. Minimum spec for the sim is 32 GB RAM + RTX 2070 + 8 GB VRAM — laptops don't clear it. |
| Teammate's incoming 3090 | **Your evaluation and submission-validation box.** 24 GB is exactly the submission inference limit, so it's the right machine to prove your policy runs under the organizers' constraints. Marginal for training a 3B VLA. |
| Jetson AGX Orin | **Not usable.** OmniGibson needs Isaac Sim on x86 with an RTX GPU. Orin is ARM. Write it off. |
| Cloud | **Where all training happens, and most evaluation.** |

**Consequence: Week 1 starts on cloud. Do not wait for the 3090 to arrive.**

### Cost estimate

> **SUPERSEDED, 2026-09-11.** The figures below are historical: they price a
> full 100-task submission and a full training run at on-demand rates, against
> a project total of $2,500–5,500. **The actual budget is $200**, all costs are
> now quoted at spot, and we are not running either. The live numbers are in
> `configs/experiments/001-dev-loop.yaml` and the 2026-09-11 revision of
> [AB_PROTOCOL](AB_PROTOCOL.md):
>
> | item | GPU-hr | spot |
> |---|---|---|
> | Our partial submission: 2 tasks × 20 instances | 11.1 | $3–6 |
> | A/B at 2×20, 1 seed, both arms | 22.3 | $7–11 |
> | *(not doing)* full 100-task submission | 776 | $233–388 |
>
> The full submission exceeds the entire budget on eval alone. Kept below
> unedited because the reasoning that led to the cut is part of the record.

Rough, to be corrected by W1-C:

- **One full evaluation pass** (2,000 rollouts, ~700–840 GPU-hr) at $0.40–0.70/hr for a 4090 ≈ **$300–600**. Parallelised 20-way, that is a bit over a day of wall-clock for the same money.
- **One full training run**: the reference recipe is 8 GPUs × 150k steps, plausibly 3–6 days on 8×A100 at ~$1.20–1.80/hr each ≈ **$700–2,000**.
- **Realistic project total including failed runs and dev time: $2,500–5,500.** (Raised: the evaluation pass costs roughly double what this section assumed, and we now have no local GPU.)

That number should drive D2. If it's out of range, scope changes — and it's much better to learn that in Week 1 than Week 5.

### Repo structure

One repo, not three. Build it around the **websocket seam** between policy and evaluator — that interface already exists in the challenge tooling, and designing to it is what makes experiments plug-and-play.

```
behavior-2026/
├── README.md
├── docker/
│   └── policy-server/Dockerfile     # submission artifact, exists from day 1
├── configs/
│   ├── robot/r1pro.yaml             # the exact file you submit
│   └── experiments/*.yaml           # one file per experiment, committed
├── policy/
│   ├── server.py                    # websocket server, model-agnostic
│   └── heads/                       # predicate / progress / failure heads
├── harness/
│   ├── launch.py                    # fan N rollouts across M workers
│   ├── worker.py
│   └── collect.py                   # gather result JSONs
├── analysis/
│   ├── parse.py                     # rollout JSON -> dataframe
│   ├── failures.py                  # failure taxonomy tagging
│   └── notebooks/
├── scripts/
│   ├── setup_cloud.sh               # reproducible box setup, idempotent
│   └── download_data.sh
└── submission/
    └── build.py                     # assembles the final zip
```

Two structural principles:

1. **`submission/build.py` exists in Week 1 and runs every week.** The submission package is not a Week 7 activity.
2. **Anything that speaks the websocket protocol is drop-in swappable.** Baseline, your model, a scripted policy, a random-action sanity check — all interchangeable behind one interface. That's what makes the harness reusable and what makes ablations cheap.

### Tooling stack

| Tool | Role | Discipline |
|---|---|---|
| **GitHub** | Code + **Issues as the only task tracker** | One repo. Don't build a parallel task system in Notion. |
| **Weights & Biases** | Every training run and every eval sweep | Non-negotiable. Both baselines already log to W&B (project `B1K`). This matters more than Notion. |
| **Notion** | Exactly three pages: **Decision Log**, **Experiment Log**, **Runbook** | Resist adding more. You're two people. |
| **Docker** | Policy server container | Build it Week 1 — it's a submission requirement, and it's how you guarantee the 24 GB constraint holds. |
| **Discord + office hours** | Support and visibility | Mondays 5–6pm Pacific. Show up. |

**The Decision Log is the highest-value Notion page.** One row per decision: date, decision, reasoning, who. In seven weeks you will absolutely re-litigate something you already settled — this is what stops that.

### Version pins and gotchas

- Clone tag **`v3.9.2`**. ⚠️ Note: the original plan says `v3.9.1`, and that *was* correct as of the 27 July announcement — the organizers bumped it since. **Check the [evaluation page](https://behavior.stanford.edu/challenge/evaluation.html) for the current tag before you clone, and re-check it before you submit.** This has already moved twice; assume it will move again.
- PyPI packages and Docker install are **temporarily unavailable** during their monorepo migration — source install only.
- First OmniGibson import takes up to ~5 minutes, once. Not a hang.
- If CuRobo fails to install, install without `--primitives`, then add CUDA 12.4 toolkit and re-run.
- If it hangs at `HydraEngine rtx failed creating scene renderer`, set `OMNIGIBSON_GPU_ID`.
- Released baseline checkpoints exist for **one task only** (`turning_on_radio`). There is no pretrained 100-task baseline to run — a real baseline number means training one yourself.

---

## 7. How you actually stand out

Ranked by return on effort:

1. **Ship the parallel eval harness as a public, documented tool.** Every team needs it. Nobody publishes one. It costs you nothing extra since you're building it anyway, and it's the most direct route to the Outstanding Open Source prize.
2. **Submit early and repeatedly.** Most teams submit once, late, and discover their packaging is broken. A leaderboard entry in Week 3 that improves weekly is a completely different risk profile.
3. **Publish the ablation.** With vs. without the predicate head, same model, same data, same eval. That's a clean result and it's the paper. It's also a real contribution regardless of where you place.
4. **Be visible in the community.** Discord and Monday office hours. The organizers notice, and the open-source prize is judged by humans.
5. **Document the failure taxonomy.** A public, data-backed breakdown of *why* VLAs fail on long-horizon household tasks is useful to everyone and costs you only the writing time — you'll have generated the data anyway.

---

## 8. Immediate next actions

- [ ] Book the 90-minute review session
- [ ] Spin up one cloud GPU today — don't wait for the 3090
- [ ] Create the GitHub repo and the three Notion pages
- [ ] Join the [Discord](https://discord.gg/bccR5vGFEx)
- [ ] Put Monday office hours in both calendars (5–6pm PT)
- [ ] Set the cloud budget ceiling

---

## Reference links

- [2026 Challenge](https://behavior.stanford.edu/challenge/index.html)
- [Evaluation and rules](https://behavior.stanford.edu/challenge/evaluation.html)
- [Submission guidelines](https://behavior.stanford.edu/challenge/submission.html)
- [Baselines](https://behavior.stanford.edu/challenge/baselines.html)
- [Dataset](https://behavior.stanford.edu/challenge/dataset.html)
- [2025 leaderboard](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html)
- [Winner's write-up (Robot Learning Collective)](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html) · [paper](https://arxiv.org/abs/2512.06951) · [code](https://github.com/IliaLarchenko/behavior-1k-solution)
- [2nd place (Comet, NVIDIA)](https://arxiv.org/abs/2512.10071) · [code](https://github.com/mli0603/openpi-comet)
