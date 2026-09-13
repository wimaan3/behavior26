# A/B protocol

**Status: frozen.** Changing anything below invalidates comparisons made under
it. Amend by adding a dated revision at the bottom, never by editing in place.

We get two, maybe three A/B cycles at ~50 GPU-hr and 2–3 days each. This
document exists so the design is chosen before the first one instead of
discovered by spending them.

> **Resolution limit, stated up front.** This design resolves **ΔQ ≥ ~0.05 on
> its two tasks**. Below that is outside its resolution, and a CI spanning zero
> means inconclusive, not "no effect". The binding constraint is **k = 2 tasks,
> not instance count** — no number of instances or seeds lowers the task-level
> floor. A result here is a claim about `set_up_a_coffee_station_in_your_kitchen`
> and `putting_shoes_on_rack`, **not about BEHAVIOR**. Full statement and the
> arithmetic: "THE DETECTION FLOOR" in revision 2026-09-11c. The response to an
> inconclusive result is pre-registered in the same revision — read it before
> the first rollout, not after the result.
>
> **OPEN DECISION, before shot one:** k=2 vs **k=4**. Widening takes the
> generalisable MDE from 0.199 to 0.042 for ~+$15 of eval, and turns "helps on
> these two tasks" into "helps on BEHAVIOR". It hinges on whether training cost
> tracks steps or tasks, which **session B measures**. Decide it there — see
> revision 2026-09-11d. Do not let this be settled by default.

---

## 0a. Revision: budget cut to $200, and the tasks changed (2026-09-05)

The A/B now runs on **two graded tasks**, not the D=1 shortlist:

| task | idx | D | rel_eval_cost | progress_offset_at_reset |
|---|---|---|---|---|
| `set_up_a_coffee_station_in_your_kitchen` | 10 | 6 | 0.594 | 0.165 |
| `putting_shoes_on_rack` | 22 | 10 | 0.733 | 0.000 |

> **Corrected 2026-09-11.** This table read `phi0_mean` 0.241 / 0.111 before the
> Jetson's relabel (`a6a5a4a`), which re-anchored progress at episode start
> instead of at the goal. The column is now `progress_offset_at_reset` and the
> numbers above are the post-relabel ones. See the 2026-09-11 revision.

**This is good news and it partly supersedes §0 below.** D=6 and D=10 mean Q is
graded, not binary, so the "no low-noise regime" argument does not apply to
these two — graded outcomes carry more information per rollout, and §2's
fallback ("if f is large, move to a graded task") has effectively already been
taken. §0 still describes the D=1 shortlist tasks accurately, and still governs
if we ever A/B on those.

Two consequences that do carry over:

- **The baseline is not zero — for one of them.** `progress_offset_at_reset` is
  what a do-nothing policy already banks, because Q is scored on the FINAL state
  and the task starts partly satisfied. Post-relabel that is **0.165 for the
  coffee station and 0.000 for the shoe rack**: ΔQ on the coffee station sits on
  top of a floor, ΔQ on the shoe rack starts from zero. They are not symmetric
  and a per-task breakdown is required when reporting.
- **σ_w must still be measured.** Graded Q reduces the variance; it does not
  remove simulator indeterminism. §2 stands, and its task choice
  (`turning_on_radio`, the only released checkpoint) is unchanged.

Costs are now quoted at **spot** rates. The harness is resumable, so a
preemption costs the in-flight rollout and nothing else.

| item | GPU-hr | spot ($0.30–0.50/hr) |
|---|---|---|
| Noise-floor measurement (§2) | 9.3 | $3–5 |
| Submission: 2 tasks × 20 public instances | 11.1 | $3–6 |
| *(not doing)* full 100-task submission | 776 | $233–388 |

**The submission ceiling is 0.020.** Q averages over all 100 tasks and 98 of
ours score zero, so even at Q=1.0 on both trained tasks the reported score is
2/100. That is the price of the budget and it is accepted deliberately: the
contribution is the A/B result, not the leaderboard position.

---

## 0. The fact that drives everything

Every `primary` task on `analysis/reward/task_shortlist.csv` has **D = 1**: one
goal predicate must flip. So per-rollout Q is **binary**, not graded. Ten of the
twelve have `n_goal_literals = 1`; only `cook_bacon` (D=7) is graded.

Two consequences:

1. **There is no low-noise regime.** Within-instance variance is `π(1−π)`,
   maximised at 0.25. An instance either reliably flips the predicate under
   simulator noise or it does not; nothing in between is available. Parameterise
   by the fraction `f` of instances that are genuine coin-flips:
   `σ_w² ≈ 0.25f`.
2. **The paired test is a discordant-pairs problem.** With binary per-instance
   outcomes, only instances where the two arms disagree carry information. The
   t-interval `analysis/compare.py` computes is a normal approximation to that,
   and it is poor when discordance is low — which is exactly our regime.

### Our earlier numbers were wrong

README claimed ΔQ ≈ 0.115 detectable unpaired and ≈ 0.033 paired at 36
rollouts. Working backwards, 0.033 at N=36 implies σ_d ≈ 0.071, i.e. a **0.5%
discordance rate** between arms. That is not a credible noise floor for a
nondeterministic simulator with binary outcomes. Treat 0.033 as an artefact of
assuming away simulator noise; the real figure is 2–6× larger. Corrected numbers
are in §1.

---

## 1. Power and cost

Generated by `analysis/power.py`, which reads the shortlist directly so it
stays correct when the Jetson regenerates it:

```bash
python analysis/power.py                    # sweep over plausible noise
python analysis/power.py --sigma-w 0.22     # once §2 has measured it
```

Model, paired, unit = one (task, instance):

```
d_i      = mean_m Q_B(i) − mean_m Q_A(i)
Var(d_i) = 2σ_w²/m + σ_b²
MDE      = (t_{0.975,N−1} + t_{0.80,N−1}) · sqrt(Var(d_i)/N),   N = k·n
```

- `σ_w` — within-instance, within-arm SD of per-rollout Q. **Simulator
  indeterminism. Unmeasured — see §2.** Shrinks with more seeds.
- `σ_b` — SD across instances of the *true* effect. Pairing removes instance
  difficulty, not effect heterogeneity, so this **does not shrink with m** and
  is a hard floor on any budget. Assumed 0.05 until the first A/B estimates it.

> **What these tables do and do not cover.** Every MDE below is a statement
> about the tasks in the design, not about BEHAVIOR. The model folds all
> heterogeneity into one per-unit `σ_b`, so its floor shrinks as `σ_b/√N` —
> correct if the heterogeneity is between *instances*, optimistic if it is
> between *tasks*. The task-level term is `σ_task²/k`, which **n does not touch
> at all**; at k=2 it is estimated from one contrast. See "THE DETECTION FLOOR"
> in revision 2026-09-11c.

Cost model: `rollout_s = scene_load + eval_timeout_frames / fps`, at 13.5 fps
and 225 s load, ×2 arms. Scene load does not scale with episode length, so a
task at `rel_eval_cost` 0.20 is roughly **0.33×** the wall clock of a 1.0× task,
not 0.20×. Tasks are always taken cheapest-first off the shortlist.

**1 seed per instance per arm** (eval GPU-hr, then MDE by noise level):

| design | N | rollouts | GPU-hr | USD | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|---|---|---|---|
| 4×3 | 12 | 24 | 3.3 | 3 | 0.147 | 0.203 | 0.284 | 0.400 |
| 4×10 | 40 | 80 | 10.9 | 11 | 0.075 | 0.104 | 0.145 | 0.204 |
| 8×5 | 40 | 80 | 12.4 | 12 | 0.075 | 0.104 | 0.145 | 0.204 |
| 8×10 | 80 | 160 | 24.8 | 25 | 0.053 | 0.073 | 0.102 | 0.143 |
| 12×5 | 60 | 120 | 24.7 | 25 | 0.061 | 0.084 | 0.118 | 0.165 |
| 12×10 | 120 | 240 | 49.4 | 49 | 0.043 | 0.059 | 0.083 | 0.116 |

**3 seeds per instance per arm:**

| design | N | rollouts | GPU-hr | USD | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|---|---|---|---|
| 4×5 | 20 | 120 | 16.3 | 16 | 0.069 | 0.091 | 0.125 | 0.174 |
| 6×5 | 30 | 180 | 26.2 | 26 | 0.055 | 0.073 | 0.100 | 0.139 |
| 8×5 | 40 | 240 | 37.1 | 37 | 0.047 | 0.063 | 0.086 | 0.119 |
| 8×3 | 24 | 144 | 22.3 | 22 | 0.062 | 0.083 | 0.113 | 0.157 |
| 12×3 | 36 | 216 | 44.5 | 44 | 0.050 | 0.067 | 0.091 | 0.126 |

### How to read it

- **Instances beat tasks per GPU-hour** when cheapest-first task selection is in
  play: 4×10 and 8×5 both cost ~11–12 GPU-hr and give the same MDE, but 4×10
  uses only the four cheapest tasks. Prefer more instances on cheap tasks until
  task-to-task generality becomes the question being asked.
- **Seeds only pay when σ_w dominates.** At f=0.05, 8×5 goes 0.075 → 0.047 for
  3× the cost — not worth it. At f=0.40 it goes 0.204 → 0.119, which is the
  difference between a usable and a useless experiment. **This is the single
  decision §2 unblocks.**
- **σ_b = 0.05 is the wall.** Even at infinite seeds and N=60, the MDE cannot go
  below ~0.018. Any hoped-for effect smaller than that is undetectable at this
  budget regardless of how it is spent.

---

## 1a. The cloud sessions

Named here because the rest of this document refers to them and the reference
has to resolve. These are the GPU sessions before the first A/B.

### Session A — first contact with a GPU

Prove the **evaluation** path runs end to end: the websocket seam, the policy
server, the real evaluator, and our harness, together, against real hardware.
Everything in this repo about that path was established by source-reading and a
mock evaluator — none of it has met a GPU.

Delivers:
- **baseline Q** on the released π₀.₅ baseline;
- **measured seconds-per-rollout** — which replaces the assumed 13.5 fps and
  225 s scene load that every cost figure in §1 currently rests on;
- **`import omnigibson` wall-clock from the volume env** — see below.

Budget: **~$25.**

#### Preflight — run BEFORE `setup_cloud.sh`

```bash
bash scripts/preflight.sh        # seconds; exits non-zero and says why
```

`setup_cloud.sh` is 30–60 minutes and pulls ~30 GB. These three checks take
seconds and each one fails for a reason that would otherwise surface *after*
that spend:

1. **`pytest tests/`** — does our own suite pass on this box at all. Green means
   the clone is intact, python works and the tool deps resolve. Red means stop
   now, not after the install.
2. **The BEHAVIOR-1K tag is `v3.9.2`.** The pinned tag moves (it was v3.9.1),
   and **`v3.9.0` is banned for evaluation** — running it means a full reinstall
   at best, a submission scored against the wrong evaluator at worst. Preflight
   refuses a banned tag, and separately confirms the tag still resolves upstream
   so a retired or mistyped tag fails in seconds rather than 40 minutes into a
   clone. If the checkout already exists, it checks what it is actually on.
3. **The robot name is `robot_r1` on both sides.** The evaluator's `r1pro.yaml`
   says `name: robot_r1`; upstream openpi's `b1k.py` says `name="robot"`. Every
   observation key is prefixed with it, so a mismatch makes `eval_b1k_wrapper`
   look up `robot::proprio` while the evaluator publishes `robot_r1::proprio` —
   **KeyError on the first step of the first rollout**, after the scene has
   loaded and the money is spent. Preflight checks our vendored yaml and that
   the openpi patch sets `ROBOT_NAME = "robot_r1"`.

**Run preflight a second time after the openpi fork is installed.** The live
openpi↔evaluator comparison
(`test_robot_name_matches_between_openpi_and_the_robot_config`) *skips* when
openpi is not importable — which is exactly the run where it would tell you
least. Before openpi exists, preflight can only confirm the patch carries the
fix; only the second run proves the first step will not KeyError.

#### Timing `import omnigibson` on the volume env

The conda env lives on the network volume (`scripts/setup_cloud.sh`). A conda
env is thousands of small files, and network storage charges per-file latency,
so the import that is seconds on local disk can be minutes on a volume. That
cost lands on every process start, so it has to be measured before we plan
around it.

**Measure it after the first import, not on it.** The first `import omnigibson`
on any pod includes a one-time shader compile the repo already documents at up
to ~5 minutes. That is not storage. And because `OMNIGIBSON_APPDATA_PATH` is
deliberately on container disk, the shader cache does **not** survive the pod —
so every new pod pays that compile again, including the A5000 one. Record it
separately as a per-pod fixed cost.

```bash
source /workspace/env.sh
time python -c "import omnigibson"        # 1st: includes the shader compile
time python -c "import omnigibson"        # 2nd: fresh process, this is the number
time python -c "import omnigibson"        # 3rd: confirm the 2nd was not a fluke
```

Report all three. The 2nd/3rd are the steady-state per-process cost.

**What it changes — less than it looks.** The import is paid **per job, not per
rollout**: `harness/launch.py :: build_jobs` groups instances into one
subprocess precisely to amortize it, and `--instances-per-job` defaults to `0`,
meaning one job per task. So a k=2 A/B cycle pays **4 imports**, not 108. Even
at 5 minutes that is 20 minutes ≈ $0.25 — irrelevant.

It only bites under parallelism, where `--workers N` with a small
`--instances-per-job` multiplies job count. So the real output of this
measurement is a ceiling on how finely we may parallelise:

```
imports per cycle = 2 arms x k tasks x ceil(n / instances_per_job)
added cost        = imports x import_seconds x $/hr / 3600
```

If the import is slow, in order of preference:

1. **Leave `--instances-per-job` at 0** (fewer, longer jobs). Costs parallelism,
   not money. This is already the default and may simply be the answer.
2. **Keep the env off the volume but still skip reinstalls**: build it at a fixed
   *container-disk* path (e.g. `/opt/behavior-env`), tar it onto the volume, and
   extract to that same path on each new pod. Identical path on every pod, so
   the not-relocatable constraint is still satisfied, and imports run at local
   disk speed. Costs one extraction per pod instead of one reinstall.
3. **Rebuild per pod** and give up the volume env entirely.

**It does not change whether to run the A5000 comparison.** Even in the worst
case — option 3, a full reinstall on the A5000 pod — that is ~30–60 min of
$0.27/hr time, call it $0.15–0.30. Choosing the cheaper card correctly saves
$8–14 *per A/B cycle* on ~30 GPU-hr of evaluation. The test pays for itself
against a full reinstall. What a slow import changes is **where the env lives**,
not whether we compare cards.

### Session B — first contact with training

Escalating: **10 steps, then 100, then one hour.** Fail fast and cheap, in that
order.

Delivers:
- does the model **load at 3.3B and fit**;
- does **`progress_loss` decrease** — the head learning at all is a
  precondition for the whole treatment, and is the manipulation check §3 of
  revision 2026-09-11c keys on;
- **peak VRAM**;
- **steps/sec for a 2-task vs a 4-task mix** — this is the **k=2 / k=4 gate**
  (revision 2026-09-11d). It is the one measurement that decides a design
  question rather than confirming a guess.
- **`action_loss` and `progress_loss` separately, at step 0 and across the first
  100 steps** — this calibrates λ. See below.

#### Calibrating λ (`progress_loss_weight`) — measurement, not intuition

The patched model computes

```
loss = action_loss + progress_loss_weight * progress_loss
```

with both terms broadcast to the same shape, so their means are directly
comparable. **`progress_loss_weight = 0.1` is a guess made before anyone had
seen either loss's scale** — and it is our one hyperparameter we cannot sweep,
since a sweep costs a training run each.

The target: the progress term should be a **meaningful but minority**
contributor — roughly **10–30% of total loss** — so it shapes the shared
representation without competing with the primary objective.

The share the progress term actually contributes is

```
share = λ·P / (A + λ·P)          A = mean action_loss, P = mean progress_loss
```

and since `A` and `P` are measured, λ is not a guess — **solve for it**:

```
λ = share·A / ((1 − share)·P)
     share=0.10  ->  λ = 0.111 · A/P
     share=0.30  ->  λ = 0.429 · A/P
```

So report `A`, `P`, `A/P`, and the share that λ=0.1 currently produces. Reading:

| measured | meaning | action |
|---|---|---|
| `A` and `P` within ~1 order of magnitude | λ=0.1 lands near the target band | keep 0.1 |
| `P` ~100× **smaller** than `A` | λ=0.1 makes the head nearly inert — the A/B would return a null **for a trivial reason**, not because the idea failed | raise λ to the solved value |
| `P` ~100× **larger** than `A` | the progress term dominates and degrades action prediction | lower λ to the solved value |

Two things to get right when reading it:

- **Confirm `progress_loss` is present at all.** If `observation.progress` is
  None the key is absent and `loss == action_loss` — the head is enabled but
  unsupervised, and every ratio below is meaningless. This is the same check as
  Branch 0 of revision 2026-09-11c §3; do it first.
- **Report the trend across the 100 steps, not just step 0.** `A` falls as the
  model learns and `P` moves as the head learns, so `A/P` drifts. A λ calibrated
  on step 0 alone can be wrong by the time it matters. If the ratio is still
  moving sharply at step 100, say so rather than pinning λ to an early transient.

This is cheap — the 100-step rung already runs — and it converts the one
un-sweepable hyperparameter we have from a guess into a measurement.

Note the asymmetry between them: session A measures things we have *assumed*
(throughput, cost); session B measures a thing we have *not decided* (k). If
only one can run, B is the one that changes what we build.

---

## 2. Measuring the noise floor — the first cloud session

Everything in §1 is a hypothesis until σ_w is measured. This is the cheapest
high-value GPU spend available and it should run **before** the first A/B.

### Design

| | |
|---|---|
| Task | `turning_on_radio` — cheapest on the shortlist (0.20×) **and the only task with a released baseline checkpoint** |
| Policy | The released `turning_on_radio` baseline. **Not** the null policy. |
| Instances | 12 **training** instances (mode: train), frozen, recorded |
| Repeats | 6 per instance, differing only in evaluator seed |
| Total | 72 rollouts ≈ **9.3 GPU-hr ≈ $9** |

### Why not one instance × N

That was the instinct and it is the wrong shape. If the single instance happens
to be deterministic — π ∈ {0,1}, which is likely given D=1 — you get 0/N or N/N
and learn only that *that instance* is stable. It does not estimate
`E[π(1−π)]` across instances, which is the quantity that sets the floor. Spread
the budget across instances instead.

### Why the baseline policy, not null

The null policy scores ~0 everywhere, so its within-instance variance is
trivially 0. Measuring it would return σ_w = 0 and be actively misleading. The
noise floor only exists where a policy is *near* the success boundary, which is
where a real arm operates.

### Why r = 6

P(observing at least one flip) at r repeats: 97% for a π=0.5 instance, 74% for
π=0.2. Below r=5 a genuinely marginal instance too often reads as deterministic.

### What to compute

```
per instance:  σ̂²_i = r/(r−1) · p̂_i(1 − p̂_i)         # unbiased for π(1−π)
pooled:        σ̂²_w = mean_i σ̂²_i        →  σ̂_w = sqrt(σ̂²_w)
also report:   #instances with ≥1 flip (a direct read on f)
```

Then re-run `python analysis/power.py --sigma-w <σ̂_w>` and pick a row from §1.

### The decision it must serve

It does not need to pin f to two decimals. It needs to answer **"is f below 0.10
or above 0.20"**, because that is the difference between 1 seed and 3 seeds
being the right design, and 3× the eval budget.

### If σ_w comes back very large (f > 0.4)

Then a per-instance binary A/B is not viable at this budget. Fall back to
`cook_bacon` (D=7, graded Q, `rel_eval_cost` 0.73) as the primary A/B task:
graded outcomes have strictly lower variance per unit of information than binary
ones. That is a design change, and it goes in a dated revision below.

---

## 3. The frozen protocol

### 3.1 Instances

- Both arms run the **identical** instance list. Non-identical coverage voids
  the pairing; `analysis/compare.py` already refuses to report on it.
- Instances are **training** instances (ids < 301). Every public-test instance
  (301–320) is scored, so iterating there is tuning on the leaderboard.
- The list is frozen in `configs/experiments/001-dev-loop.yaml` and changes only
  by dated revision here. **Changed twice: revision 2026-09-11b took it from
  `[10, 11, 12]` to n=20, and 2026-09-11c to n=27 (instances 10–36).**

### 3.2 Seeds and passes

A **pass** is one complete evaluation of every (task, instance) at one rollout
each — exactly what the organizers replicate. Run `m` passes per arm, differing
only in evaluator seed.

- `m` is chosen from §1 **before the run** and recorded in the run manifest.
- Seeds are recorded. A pass is never discarded.

### 3.3 What is reported

Never a best run. Specifically:

- **Headline: the median pass Q**, per arm. This is the statistic the organizers
  can replicate, and the median is robust to one anomalous pass.
- **Spread: min–max across passes**, always printed next to the headline. A
  headline without a spread is not a result.
- **Effect: paired ΔQ at the (task, instance) level**, using the per-instance
  mean over passes, with a 95% CI.
- **n and the pre-registered design**, so a reader can check the achieved N
  matches what was planned.

Reporting the best pass, or re-running one arm and keeping the better result, is
the single fastest way to a score the organizers cannot replicate. Non-replicable
scores get bumped out of the top 5.

### 3.4 Stopping rule

- The design (k, n, m) is fixed **before** the first rollout and written into the
  run manifest.
- **Stop when the pre-registered rollouts finish.** No extending because the
  result is nearly significant — optional stopping inflates the false-positive
  rate well above 5%, and we have too few cycles to spend one on a phantom.
- CI excludes 0 → adopt, and report the CI.
- CI includes 0 → **inconclusive, not "no effect."** Record the point estimate
  and the achieved MDE. Spending a second cycle on the same question is only
  justified if the point estimate exceeds the MDE the *next* design could reach.
  **The branch to take is pre-registered** — manipulation check first, then one
  of {larger λ / more instances / a different task pair} depending on where the
  point estimate falls and whether the two tasks agree. See revision
  2026-09-11c §3. Decided before the number existed; do not re-litigate it after.
- A third cycle on the same question is not available. Choose accordingly.

### 3.5 Both arms train from the same commit — hard rule

**Both arms must be trained from the same commit of this repo and the same
commit of the openpi fork.** The only permitted difference is the treatment
itself (the config name).

See also **§3.6**, which is the same rule applied to the training data rather
than the code.

If a bug is found after arm A trains and is fixed before arm B, **the comparison
is void** — the measured ΔQ is the sum of our contribution and the bugfix, and
nothing separates them. Options are: re-train arm A at the new commit, or defer
the fix to after the cycle. There is no third option.

Record in every run manifest:

```
behavior26_commit   <sha>          # must match across arms
openpi_commit       <sha>          # must match across arms
openpi_patch_sha256 <sha256>       # training/patches/*.patch
config              pi05_b1k_frozen_vlm | pi05_b1k_frozen_vlm_progress
dataset_root        <path>         # must match across arms
progress_key        <column|null>
seeds               [...]
design              k=<> n=<> m=<>
```

---

### 3.5a Thread configuration — tag every number, and do NOT read a dev/stock split as invalidating the A/B

Torch's thread configuration is a free variable that changes throughput and can,
in principle, change numbers. `evaluator.py` (v3.9.2) ships the disabled remains
of a known fix — `TORCH_NUM_THREADS = None`, `TORCH_NUM_INTEROP_THREADS = None` —
so torch sizes both pools from **detected cores**, not the cgroup quota: measured
**192/192 against a 20.4-core quota** on the session-A pod. A participant traced
3 fps to contact-cache thread thrashing and reported 20–30 fps with
`set_num_threads(4)` + `set_num_interop_threads(1)`.

**Every measured number carries the thread configuration it was produced under.**
`scripts/first_rollout.sh` records `intra_threads` and `inter_threads` in each
result row for exactly this reason. A seconds-per-rollout or Q figure without
that tag is not usable.

Conditions, as named in the results:

| | intra-op | inter-op | how |
|---|---|---|---|
| **a** stock | detected cores | detected cores | nothing set |
| **b** | 4 | detected cores | `OMP_NUM_THREADS=4` — reaches the intra-op pool ONLY |
| **c** | 4 | 1 | `PYTHONPATH=scripts/threadfix` + `BEHAVIOR_TORCH_THREADS/INTEROP` |

#### If the fast config changes the numbers, the A/B is still valid

Suppose actions or trajectories differ between stock and fast, and we therefore
run **dev iteration under fast and reported numbers under stock**. It is easy to
read that split as compromising the comparison. **It does not.**

ΔQ is a *paired* difference and the thread configuration is *constant across
arms*. Both arms run the same config, so whatever offset it induces is common to
both and cancels in `d_i = Q_B(i) − Q_A(i)`. The A/B remains valid measured
entirely under the fast config.

What the split does cost is narrower and worth stating exactly: **we lose the
ability to predict submission Q from dev Q.** The dev number is a valid estimate
of the *effect*; it stops being a reliable estimate of the *level* the organizers
will reproduce, because they replicate under stock. So:

- **A/B conclusions** — adopt/reject the progress head — stand on fast-config
  numbers. No re-run needed.
- **Any absolute Q we report** — submission, or a claim about baseline level —
  must be produced under stock config.
- Do not re-run a completed A/B "because it was measured under the fast config".
  That is the mistake this section exists to prevent.

#### Where divergence could actually come from

Not the model forward: `set_num_threads` governs CPU intra-op parallelism, and a
GPU forward pass has its reduction order set by CUDA kernels regardless. The live
path is CPU-side work before the GPU — concretely `evaluator.py:344-351`, which
does per-step `relative_pose_transform` / `th.cat` / `mat2pose` over camera poses
and puts the result in `obs[...::cam_rel_poses]`, where the policy sees it.

Measured with `scripts/thread_numerics_probe.py`: ops of that shape (4×4 pose
math, concatenation) are **bitwise identical** across thread counts — they are
below torch's parallelization grain size. Only full reductions over millions of
elements move. That points toward the favourable branch, but it was measured at
intra 1-vs-3 on a 6-core box; 192-vs-4 is a different regime and the pod
measurement governs.

And `OMP_NUM_THREADS` is a **general OpenMP variable** — other libraries in the
Isaac Sim stack may read it. So torch being bit-identical does not close the
question. If actions match but the trajectory still moves, the difference is
physics-side, and that is the one that breaks replication. `first_rollout.sh`
therefore records `sim_steps`, `steps`, `agent_distance.{base,left,right}`,
`normalized_agent_distance` and `normalized_time` per condition: they move
continuously, so they expose a divergence that a binary Q on `turning_on_radio`
would hide entirely.

---

### 3.6 Both arms train on the identical filtered dataset — hard rule

**Same status as §3.5: violating this voids the comparison.**

The label pipeline cannot soundly label every episode. It labels what it can and
leaves the rest out of the sidecar — **5 of 200 for `putting_shoes_on_rack`,
1 of 200 for `set_up_a_coffee_station_in_your_kitchen`** (`valid_rate` 0.975 and
0.995 on the shortlist).

Arm B trains on the labelled set. Arm A needs no labels, so the obvious thing is
to point it at the pristine 200-episode root — and that is exactly the mistake.
The arms would then differ by **~2.5% of the training data as well as by the
head**, and the measured ΔQ would be the sum of the treatment and a data-volume
difference, with nothing separating them. It is the §3.5 failure in a different
coat.

**The rule: both arms train on the SAME filtered root. Arm A simply does not map
the progress column.**

```
python scripts/merge_progress_labels.py \
    --dataset-root ~/data/b1k/<task> \
    --labels ~/labels/<task>.parquet \
    --out-root ~/data/b1k/<task>+progress \
    --drop-unlabelled

# arm A (baseline)   --data.base_config.dataset_root=<out-root>
# arm B (treatment)  --data.base_config.dataset_root=<out-root> --data.progress_key=progress
```

`--drop-unlabelled` removes every **episode** containing an unlabelled frame —
episode granularity, not frame, because a half-labelled episode is unsound
either way and the model sees episodes. It writes
`meta/progress_filter.json` recording exactly which episodes went, which is what
makes this auditable once the run is over. Without that manifest, "both arms saw
the same data" is a claim nobody can check afterwards.

The two alternatives are both worse and are rejected deliberately:

| option | why not |
|---|---|
| Fabricate labels for unlabelled frames (`--allow-missing --missing-fill`) | Trains the head on invented targets. Keeps the arms symmetric but corrupts the treatment — worse than not training the head at all, and the flag says so. |
| Let arm A use the pristine root | The asymmetry above. This was the script's *stated design intent* until 2026-09-11 ("the baseline arm provably reads the original bytes") — a virtue that was actually the bug. |

Add to the §3.5 run manifest, for both arms:

```
dataset_root          <path>      # must match across arms -- the FILTERED root
progress_filter_sha256 <sha256>   # of meta/progress_filter.json; must match
n_train_episodes      <int>       # must match across arms
```

Note that episode indices are **not** renumbered by the drop. `videos/` is
symlinked back to the pristine root and keyed by the original episode index;
renumbering would misalign every frame with its video, silently, because the
shapes would still be right.

---

## 4. Gaps in `analysis/compare.py`

It implements the coverage guard and the paired t-interval, and nothing else in
§3. Not written yet, per instruction — this is the list.

| # | Gap | Why it matters |
|---|---|---|
| 1 | ~~**Joins on `rollout_id`**~~ — **FIXED 2026-09-11** (`66d365d`) | Pairs A's rollout 3 with B's rollout 3. Under nondeterminism there is no correspondence between them. Now aggregates seeds per (task, instance) within each arm, then pairs on (task, instance); `PAIR_KEYS` no longer contains `rollout_id`. This was the one that produced a wrong number rather than a missing one. |
| 2 | Reports **mean** ΔQ only | §3.3 requires the median pass and its spread. No pass-level concept exists in the code. |
| 3 | No per-pass decomposition | Cannot produce "median pass Q, range [a,b]" for either arm — the headline number the protocol requires. |
| 4 | No provenance check | Nothing verifies both arms share a commit, dataset root, or patch hash. §3.5 is currently a rule with no enforcement. |
| 5 | Continuous-Q assumption | t-interval on `d_i`. With D=1 the outcome is binary and the exact test is McNemar-shaped; the normal approximation is weakest at low discordance, which is our regime. At minimum report the discordant-pair counts. |
| 6 | `min_detectable_dq` is post-hoc | Computed from observed SE. Useful as a diagnostic, but there is no check that achieved N matches the pre-registered design (§3.4). |
| 7 | No stopping-rule record | Nothing carries the pre-registered design, so nothing can flag that a run was extended. |

Gap 1 was a correctness bug under the protocol and is fixed. 2–4 are missing
features; 5–7 are rigour. Gap 5 is substantially defused for the two tasks we
actually run — see the 2026-09-11 revision — but still stands for the D=1 tier.

---

## Revisions

### 2026-09-11d — k=4 becomes a decision BEFORE shot one, not a consolation after it

Revision 2026-09-11c filed "widen to k=4" under Branch 3, i.e. as a response to a
disappointing result. **That was the wrong place for it.** k=2 → k=4 is the
largest single improvement in claim strength available to this project, and a
design choice that valuable must be made before the first rollout, not
discovered after a weak one. Branch 3 still stands as a *fallback*; this
revision adds the decision point that should come first.

#### Why it is the biggest lever we have

From "THE DETECTION FLOOR" in 2026-09-11c, the task-level floor at σ_task = 0.02:

| | k=2 | k=4 |
|---|---|---|
| generalisable MDE (σ_task=0.02) | **0.199** | **0.042** |
| within-task MDE at f=0.20, n=27 | 0.049 | 0.036 |
| eval, both arms, n=27 | 30.1 GPU-hr, $9–15 | 68.3 GPU-hr, $20–34 |

That is the difference between **"helps on these two tasks"** and **"helps on
BEHAVIOR"**, for **+38.2 GPU-hr ≈ +$11–19** of evaluation. Nothing else on the
table moves the claim that far for that little. The within-task MDE improving as
a side effect (0.049 → 0.036) is a bonus, not the point.

#### The open question: does training cost track steps, or tasks?

The above is the *evaluation* cost, and it is small. The decision hinges on
**training** cost, which is not yet measured.

The hypothesis is that it is roughly flat: π₀.₅ trains multi-task, so four tasks
at the same step budget is **the same number of gradient steps on more diverse
data** — not twice the training. If that holds, k=4 costs ~nothing in training
and +$11–19 in eval, and it should simply be taken.

It is a hypothesis and it can fail two ways:
1. **Throughput** — a four-task mix may cost more wall-clock per step
   (dataloader, more norm-stat groups, worse locality).
2. **Convergence** — four tasks may need *more steps* to reach the same loss,
   in which case cost tracks tasks after all and the trade changes.

#### DECISION POINT: after session B, before shot one

Session B (the training-throughput session) measures exactly this. **Decide k=2
vs k=4 there, on the evidence, before any A/B rollout.** Do not defer it into
Branch 3.

What session B must report for this decision to be makeable:

| measurement | why |
|---|---|
| steps/sec on a 2-task mix vs a 4-task mix, same config | isolates throughput |
| `action_loss` and `progress_loss` vs step for both mixes | isolates convergence — does 4-task need more steps for the same loss? |
| peak memory for both mixes | a 4-task mix that does not fit is a different conversation |

**Pre-committed decision rule**, so this is not re-argued with the numbers in
hand:

- **Throughput within ~10% and steps-to-target-loss within ~25% → cost tracks
  STEPS. Take k=4.** The eval delta is $11–19 and the claim strength roughly
  quintuples.
- **Four tasks need materially more steps, or throughput drops sharply → cost
  tracks TASKS.** Re-price against the $200 budget and default to k=2, recording
  the measured training delta so the choice is auditable.
- **Doesn't fit in memory → k=2**, and note it as a hardware constraint rather
  than a design preference.

#### The 3rd and 4th tasks, if k=4 is taken

- **3rd: `outfit_a_basic_toolbox`.** Rank 2 on the graded tier, already
  measured. `frac_intermediate` 0.811, `census_alignment` aligned, and the best
  demo-completion rate of the top four (0.79) with the lowest never-credited
  count (0.265). The strongest available addition on every axis.
- **4th: `preparing_lunch_box`** — and **not** `thawing_frozen_food`, despite
  the latter ranking higher on `gradient_score` (0.682 vs 0.600). Our own
  defect report lists `thawing_frozen_food` among tasks where the reward may be
  under-firing: **18 of 200 demos reach the goal** and 1.84 units are never
  credited, with `census_alignment` ambiguous. A task whose goal is largely
  unreachable in the demo corpus adds a task-level draw near the floor — it
  would inflate σ_task rather than help estimate it, which is the opposite of
  why we are widening. `preparing_lunch_box` is aligned, 0.525 demo completion,
  0.665 never-credited.

#### Two things adopting k=4 breaks — fix them in the same change, not after

1. **`outfit_a_basic_toolbox` is currently the RESERVE** in
   `001-dev-loop.yaml`. Promoting it to a run task leaves the reserve slot
   empty. **DECIDED 2026-09-11: leave it empty, and say so.** The next graded
   candidates are `setting_the_fire` and `put_together_a_basic_pruning_kit`, and
   neither has been justified the way the four run tasks have — naming one
   purely to keep the slot occupied would dress an unexamined choice as a
   considered one. An empty reserve is an honest statement that we have not
   picked a fallback; a back-filled one is a decision nobody made.
2. ~~**`test_every_task_is_genuinely_graded` asserts `frac_intermediate > 0.75`,
   and `preparing_lunch_box` is 0.721.**~~ **DONE 2026-09-11 — restated to 0.5,
   on the concept rather than on the sample.** The test is asking "is Q graded
   here, or a step function wearing a D>1 label?", because `power.py` models
   `σ_w = 0.5·√(f·D)·step` and that understates σ_w by ~√D on a task that really
   jumps 0 → 1. 0.5 is the line for three reasons, none of them "it fits our
   tasks": it is the label author's own calibration (`labels.py`: "0 for a step
   function, ~0.5+ for a genuine staircase"); it has a meaning in our regime,
   where Q is scored on the final state and a timing-out policy finishes on a
   partial level more often than not above 0.5; and **nothing real sits near
   it** — across every eligible task `frac_intermediate` is either exactly 0.000
   (13 tasks) or ≥ 0.558 (9 tasks), so 0.5 lies in an empty band and its exact
   value changes no verdict. The old 0.75 was calibrated on a two-task sample.
   A test now fails if any eligible task ever lands near the cutoff, so a
   borderline case forces the number to be re-argued instead of silently decided.

Neither was a reason not to take k=4, and both are now settled ahead of it: the
threshold is restated, and the reserve will be left deliberately empty. The k=4
edit itself is still gated on session B.

---

### 2026-09-11c — n=27; the detection floor, stated; and what we do if the CI spans zero

Three things: the instance list moves again, this design's resolution limit is
written down as a limitation rather than left to be derived, and the response to
an inconclusive result is fixed **now, before any number exists**.

#### 1. n = 20 → 27 (instances 10–36)

Revision 2026-09-11b took n to 20, which reaches MDE 0.058 at f=0.20 — short of
the 0.05 we care about at pessimistic noise. **n=27 reaches 0.049**, so the
design clears its target across the whole plausible noise range rather than only
the optimistic end.

| n | N | GPU-hr | $ spot | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 3.3 | $1–2 | 0.109 | 0.137 | 0.181 | 0.245 |
| 20 | 40 | 22.3 | $7–11 | 0.035 | 0.044 | 0.058 | 0.078 |
| **27** | **54** | **30.1** | **$9–15** | **0.030** | **0.037** | **0.049** | **0.067** |
| 54 | 108 | 60.1 | $18–30 | 0.021 | 0.026 | 0.034 | 0.047 |

Same contiguity and prefix reasoning as 2026-09-11b: ids 10–36, a block, with
`[10, 11, 12]` still the prefix. All 27 verified present for both tasks and for
the `outfit_a_basic_toolbox` reserve.

---

#### 2. THE DETECTION FLOOR — read this before reading any result

**This design resolves ΔQ ≥ ~0.05, for these two tasks. Below that is outside
its resolution, and the binding constraint is k=2 tasks, not instance count.**

That second clause is the part that is easy to get wrong, so here it is
explicitly. Write the per-instance difference as

```
d_ti = μ + τ_t + ε_ti          τ_t ~ (0, σ_task²)   between TASKS
                               ε_ti ~ (0, σ_inst²)  between instances, within task

Var(d̄) = σ_task²/k  +  σ_inst²/(k·n)  +  2σ_w²/(k·n·m)
         └─ does NOT shrink with n ─┘
```

Only the first term matters here, and **n is absent from it.** Adding instances
drives the second and third terms down and leaves the first exactly where it
was. At k=2 that term is σ_task²/2, permanently.

What that costs, as a minimum detectable ΔQ at *any* n and *any* m:

| σ_task | k=2 | k=4 | k=8 | k=12 |
|---|---|---|---|---|
| 0.02 | **0.199** | 0.042 | 0.023 | 0.018 |
| 0.03 | **0.299** | 0.062 | 0.035 | 0.027 |
| 0.05 | **0.498** | 0.104 | 0.058 | 0.044 |

The k=2 column is brutal because a task-level effect estimated from two draws
carries one degree of freedom, and t(1, 0.975) = 12.7. **No amount of instances
or seeds moves that column.**

So there are two different claims available from this experiment, and they have
very different support:

- **"The progress head helps on these two tasks."** Supported. MDE 0.049 at
  f=0.20, N=54. This is what §1's tables describe, and it is a real result.
- **"The progress head helps on BEHAVIOR."** **Not supported, at any budget we
  can reach with k=2.** We would be generalising from two draws.

**`analysis/power.py` reports the first, not the second.** Its model folds all
heterogeneity into a single per-unit σ_b = 0.05, so its MDE shrinks as
σ_b/√N — which is right if effect heterogeneity is between *instances* and
optimistic if it is between *tasks*. With k=2 we cannot tell the two apart: the
data gives one task-level contrast. Treat every σ_b-derived floor in §1 and in
the 2026-09-11 revision as conditional on heterogeneity being instance-level.

Consequences to carry into the write-up:

- Report ΔQ **per task** as well as pooled. If the two tasks disagree in sign or
  size, the pooled number is hiding the finding, and σ_task is not small.
- Never write "on BEHAVIOR" or "generalises" about a k=2 result. Write "on
  `set_up_a_coffee_station_in_your_kitchen` and `putting_shoes_on_rack`".
- An effect below ~0.05 is **not a null result** — it is outside our resolution.
  §3.4 already says a CI spanning zero is inconclusive rather than "no effect";
  this is the quantitative reason.
- If the goal becomes a claim about the benchmark, the next spend is **more
  tasks**, not more instances. k=4 at n=27 costs roughly double and buys a real
  task-level column; n=54 at k=2 costs the same and buys none.
  **This is now a live decision scheduled before shot one, not a contingency —
  see revision 2026-09-11d.**

---

#### 3. PRE-REGISTERED: what we do if the CI spans zero

Fixed before the first rollout, per §3.4. We have two, maybe three cycles, and
§3.4 forbids a third on the same question — so at most one of the branches below
is ever taken. **Check them in order; the first that matches decides.**

**Branch 0 — manipulation check, before anything else.**
Was the treatment actually delivered? `compute_losses` returns
`{action_loss, progress_loss, loss}` and `train_b1k.py` logs them separately,
precisely so this is answerable. If `progress_loss` is flat, NaN, or absent in
arm B's logs, the head never learned and **this is not a null result — it is a
broken run.** Fix and re-run the same design. Does not consume the question.

**Branch 1 — point estimate below the floor (|ΔQ| < 0.02).**
The effect, if any, is smaller than this design can ever resolve: reaching 0.02
needs n=159 at f=0.20, and the instance-level floor at N=54 is 0.019 even at
infinite seeds. **Do not buy more instances — they cannot get there.** Take one
of:
  - **larger λ**, if Branch 0 shows the head learning but weakly. Current
    `progress_loss_weight = 0.1` in `pi05_b1k_frozen_vlm_progress`. Raise it and
    re-run; this tests a *stronger treatment*, not the same one again.
  - otherwise **stop** and report the interval honestly as a bounded null on
    these two tasks. That is a publishable result and it costs no cycle.

**Branch 2 — point estimate in 0.02–0.05, CI spanning zero, both tasks agreeing
in sign.** Under-powered, not absent. This is the only branch where **more
instances** is correct. n=54 reaches 0.034 at f=0.20 for $18–30. Pre-committed
rule: spend the cycle only if the observed |ΔQ| exceeds the MDE the *next*
design would reach — i.e. only if |ΔQ| > 0.034. §3.4 requires that test and it
is not optional.

**Branch 3 — the two tasks disagree in sign, or differ by more than the pooled
|ΔQ|.** σ_task is large and the pooled estimate is meaningless. More instances
would sharpen a number that is not worth sharpening. Take **a different task
pair**, or widen to k=4. This is the only branch that buys generality.
**But k=4 should already have been decided before shot one — see revision
2026-09-11d, which moves it to a decision point after session B.** If it was
considered and declined there, this branch is where that decision gets revisited
with evidence; it should not be the first time the option is raised.
(Candidates and the reason to prefer `preparing_lunch_box` over
`thawing_frozen_food` are in 2026-09-11d.)

**Not permitted, in any branch:** extending the run because the result is nearly
significant, re-running one arm and keeping the better number, or reporting the
best pass. §3.3 and §3.4 already forbid these; naming them here removes the
temptation at the moment it will actually be felt.

---

### 2026-09-11b — the frozen instance list goes from n=3 to n=20

**This changes §3.1's frozen list.** `configs/experiments/001-dev-loop.yaml`
now runs instances **10–29** instead of **[10, 11, 12]**.

> **Superseded by 2026-09-11c**, which took n to **27** (ids 10–36) so the
> design clears 0.05 at pessimistic noise too. The reasoning below stands; only
> the number changed.

#### Why the freeze does not protect anything here

§3.1 freezes the list so numbers stay comparable across iterations. **There are
no prior iterations.** Nothing has run on a GPU (see README status), so there is
no measurement the freeze is protecting — the rule is live, but its subject is
empty. Changing the list now costs nothing; changing it after the first A/B
would cost that cycle.

#### Why n=3 had to go

N=6 gives an MDE of **0.181**. An auxiliary loss on a frozen VLM plausibly lands
in the **0.02–0.05** range. The design could not have detected its own
hypothesis: we would have spent a cycle and learned nothing, and the CI would
have spanned zero whatever happened. At n=20, N=40 and the MDE is **0.058** at
f=0.20 — a 3.1× improvement at every noise level, for **22.3 GPU-hr / $7–11**
on spot. Against a $200 budget that is the cheapest real gain available.

| | n=3 (N=6) | n=20 (N=40) |
|---|---|---|
| MDE at f=0.05 | 0.109 | **0.035** |
| MDE at f=0.10 | 0.137 | **0.044** |
| MDE at f=0.20 | 0.181 | **0.058** |
| MDE at f=0.40 | 0.245 | **0.078** |
| eval cost, both arms | 3.3 GPU-hr | 22.3 GPU-hr ($7–11) |

#### What n=20 does and does not buy — state this when reporting

It covers the **top** of the 0.02–0.05 target range, not the bottom:

- At f ≤ 0.10, MDE ≤ 0.044 — the 0.05 end is comfortably in reach.
- At f = 0.20, MDE is 0.058, so **0.05 is not quite reached**; that needs n=27.
- **0.02 is out of reach at two tasks, at any n we can afford** — it needs n=159
  at f=0.20. And the σ_b floor at N=40 is **0.0227** even at infinite seeds, so
  an effect of 0.02 is not detectable on this design however the budget is spent.

n needed to hit a target MDE (2 tasks, 1 seed):

| target | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|
| 0.06 | 8 | 12 | 19 | 34 |
| 0.05 | 11 | 16 | 27 | 48 |
| 0.04 | 16 | 24 | 41 | 74 |
| 0.02 | 59 | 93 | 159 | 292 |

n=20 was chosen as instructed. **n=27 would secure the 0.05 target even at
pessimistic noise** for roughly $9–15 instead of $7–11; worth considering before
the first cycle, and it is a cheaper change now than later.

#### The instances exist — verified, not assumed

`resolve_instance_ids` returns train ids **unvalidated** (`evaluator.py`
l.75–76), so neither the evaluator nor our own `< 301` guard would catch an id
that does not exist. It would fail in `Evaluator.load_task_instance` with
`FileNotFoundError`, on a rented GPU, partway through a sweep.

Enumerated from `2026-challenge-task-instances.zip` (HF `behavior-1k/zipped-datasets`),
counting the `*-tro_state.json` files the evaluator actually loads:

- **All 100 tasks ship exactly 300 training instances.**
- The range is **task-dependent**: 50 tasks are ids **0–299**, 50 are **1–300**.
  Both dev-loop tasks — and the `outfit_a_basic_toolbox` reserve — are 0–299, so
  ids 10–29 all exist.
- Our `< 301` guard is therefore correct and not off by one: 300 is a genuine
  training instance on half the corpus. But **id 0 is not universally safe**, and
  a future config drawn from a different task must re-check its range.

The list is kept contiguous, with the old `[10, 11, 12]` as its prefix, so any
smoke run already done on those three sits inside the new set rather than beside
it. A test now pins n=20, contiguity, and the 0–299 bound.

---

### 2026-09-11 — power re-run on the post-relabel shortlist; §1 superseded for the graded tier

The Jetson's relabel (`a6a5a4a`, `d3b1769`) is merged. It re-anchors progress at
episode start and adds a **`graded` tier** to `task_shortlist.csv`, which is
where both tasks we actually run now live. §1's tables were generated against
the `primary` tier under the assumption that every task is D=1 and Q therefore
binary. **That assumption does not hold for our design, and §1 is superseded for
it.** §0 and §1 continue to govern the D=1 tier.

#### What changed in the model

`analysis/power.py` no longer assumes binary Q. Q moves in steps of one credit
unit, so with `f` the fraction of a task's `D` goal units that are coin-flips:

```
σ_w = 0.5 · sqrt(f · D) · step        step = measured max_step_frac
```

At D=1, `step` = 1 and this is exactly the old `σ_w² = 0.25f`. At D>1, `step` ≈
1/D, so σ_w falls as **sqrt(D)** — a flipped unit moves Q by 1/D, not by the
whole thing. The measured `max_step_frac` agrees with 1/D to within 2% on both
tasks, i.e. the goal predicates carry equal weight.

Grading is only real if episodes occupy the intermediate levels, and here they
do — `frac_intermediate` is **0.81** (coffee station) and **0.80** (shoe rack).
A nominally graded task with a low `frac_intermediate` is binary in disguise;
`power.py` now prints the figure and flags that case.

| | binary (D=1) | graded (our two) | |
|---|---|---|---|
| σ_w at f=0.05 | 0.112 | 0.041 | 2.7× lower |
| σ_w at f=0.20 | 0.224 | 0.082 | 2.7× lower |
| σ_b share of Var(d_i) at f=0.20, m=1 | 2.4% | 15.6% | |
| σ_b share of Var(d_i) at f=0.05, m=1 | 9.1% | 42.4% | |
| MDE at N=40, m=1, f=0.20 | 0.145 | **0.058** | 2.5× better |

#### §0's premise, rechecked

§0 opens "Every `primary` task has **D = 1**". Post-relabel that is no longer
literally true — the tier now also carries `cook_bacon` (D=7), `wash_dog_toys`
(D=6) and `make_pizza` (D=2). **Its conclusion survives anyway**, because two of
those three are graded in name only:

| primary task, D>1 | D | frac_intermediate | gradient_score |
|---|---|---|---|
| `cook_bacon` | 7 | 0.591 | 0.084 |
| `wash_dog_toys` | 6 | **0.000** | 0.000 |
| `make_pizza` | 2 | **0.000** | 0.000 |

`wash_dog_toys` and `make_pizza` never occupy an intermediate level: they jump
0 → 1 and behave exactly like binary tasks despite D>1. Treat the primary tier
as binary, as §0 says — but on the measurement, not on D. `power.py` now prints
`frac_intermediate` and flags this case rather than trusting D.

#### The re-run tables

`python analysis/power.py --tasks-from configs/experiments/001-dev-loop.yaml`
(new flag — the table now describes the design we actually froze, rather than
the cheapest k off a tier we do not run):

**1 seed per instance per arm:**

| design | N | rollouts | GPU-hr | USD | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|---|---|---|---|
| 2×3 | 6 | 12 | 3.3 | 2 | 0.109 | 0.137 | 0.181 | 0.245 |
| 2×5 | 10 | 20 | 5.6 | 3 | 0.076 | 0.096 | 0.126 | 0.171 |
| 2×10 | 20 | 40 | 11.1 | 6 | 0.051 | 0.064 | 0.084 | 0.114 |
| 2×20 | 40 | 80 | 22.3 | 11 | 0.035 | 0.044 | 0.058 | 0.078 |

**3 seeds per instance per arm:**

| design | N | rollouts | GPU-hr | USD | f=0.05 | f=0.10 | f=0.20 | f=0.40 |
|---|---|---|---|---|---|---|---|---|
| 2×3 | 6 | 36 | 10.0 | 5 | 0.086 | 0.098 | 0.119 | 0.153 |
| 2×5 | 10 | 60 | 16.7 | 8 | 0.060 | 0.069 | 0.083 | 0.107 |
| 2×10 | 20 | 120 | 33.4 | 17 | 0.040 | 0.046 | 0.055 | 0.071 |
| 2×20 | 40 | 240 | 66.8 | 33 | 0.027 | 0.031 | 0.038 | 0.049 |

#### What it changes about the design

- **The discordant-pairs framing (§0.2) mostly dissolves.** With D=6 and D=10
  and ~80% of each episode at intermediate progress, per-instance outcomes are
  not binary, so "only instances where the arms disagree carry information" no
  longer describes our regime. Gap 5 in §4 is defused for these two tasks and
  still stands for the D=1 tier.
- **Instances beat seeds, decisively.** 2×10 at 1 seed (11.1 GPU-hr) and 2×5 at
  3 seeds (16.7 GPU-hr) reach the *same* MDE of ~0.084 at f=0.20 — the seeded
  design costs 1.5× for nothing. `n` cuts both variance terms; `m` cuts only
  `2σ_w²/m` and never touches `σ_b²`. **Spend the eval budget on instances.**
- **§2's noise-floor measurement is no longer decisive.** Its stated job was to
  answer "is f below 0.10 or above 0.20", because that was the 1-seed/3-seed
  decision. On graded tasks 1 seed is the answer across that whole range, so §2
  now only calibrates *how much* to believe the MDE, not what design to buy. It
  is still worth its $3–5, but it no longer blocks the first A/B.
- **n = 3 is too few and it is the binding constraint.** The frozen instance
  list is `[10, 11, 12]`, giving N=6 and an MDE of **0.181** at f=0.20 — larger
  than the 1st-to-5th place gap the pairing was introduced to resolve. Going to
  n=20 costs 22.3 GPU-hr / **$11** and reaches 0.058. Changing the list is a
  protocol change and needs its own dated revision; this one only records that
  the current list cannot answer the question.
  **→ Superseded by revision 2026-09-11b, which made the change: the list is now
  instances 10–29.**

#### Costs, recomputed at spot

Recomputed from `analysis/reward/full_corpus_lengths.csv` (all 100 tasks) under
the §1 cost model:

| item | rollouts | GPU-hr | $0.30/hr | $0.50/hr |
|---|---|---|---|---|
| Submission: 2 tasks × 20 public instances | 40 | 11.1 | $3.34 | $5.57 |
| A/B at the recommended 2×20, 1 seed, 2 arms | 80 | 22.3 | $6.69 | $11.15 |
| Noise-floor measurement (§2) | 72 | 9.3 | $2.79 | $4.65 |
| *(not doing)* full 100-task submission | 2,000 | 776 | **$233** | **$388** |

The full submission is not merely expensive, it **exceeds the entire $200
project budget even at the cheapest spot rate** — $233 of eval alone, before any
training. That is the arithmetic that settles it, and it is firmer than the
~$450 working figure it replaces.

**The 2-task ceiling is 0.020, and the full submission would score ~0.025.** The
ceiling is arithmetic: Q averages over 100 tasks and 98 of ours are zero, so
2/100 even at Q=1.0 on both. The ~0.025 is what the full run would be expected
to *achieve*. Spending $233–388 more to move 0.020 → 0.025 is not a trade this
budget can make, and it is not what the contribution rests on.

#### Also in this revision

- `analysis/compare.py` gap 1 is fixed (`66d365d`): pairing is now on
  (task, instance) with seeds averaged within each arm, which is the `d_i` this
  document's model has always specified. Before the fix, `compare.py` and
  `power.py` were describing different quantities.
- `analysis/power.py` gains `--tasks-from <config.yaml>`, so the power table and
  the frozen config cannot drift apart.
- The §0a task table's `phi0_mean` column was corrected in place to
  `progress_offset_at_reset`; the relabel redefined the quantity, so the old
  values were wrong rather than merely stale. The shoe rack's offset is **0.000**
  — a do-nothing policy banks nothing there, unlike the coffee station's 0.165.

