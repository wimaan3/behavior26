# behavior26

Predicate-aware VLA for the [2026 Stanford BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html).

**Deadline: 16 October 2026.**

---

## What this repo is

Infrastructure for running, measuring and packaging BEHAVIOR evaluations, plus our
contribution on top of a π0.5 baseline.

It is built around the **websocket seam** that already exists between policy and
evaluator. Anything that speaks that protocol is drop-in swappable — baseline, our model,
a scripted policy, a random-action sanity check. That is what makes ablations cheap.

```
policy/     the model + websocket server (what the evaluator talks to)
training/   openpi fork patches + the progress head (our contribution)
harness/    parallel rollout orchestration — the thing that makes iteration possible
analysis/   rollout JSON -> dataframe -> failure taxonomy -> paired A/B
submission/ package + validate the final zip
tests/      mock evaluator + regression tests — the whole pipeline, no GPU
configs/    one committed YAML per experiment; robot config
scripts/    cloud setup, data download, first-rollout timing
docker/     policy server image (a required submission artifact)
docs/       project charter
```

## Testing the pipeline without a GPU

GPU time is the scarcest resource here, and a bug found on a rented card is a bug paid
for twice. Two pieces make the entire chain runnable on a laptop in seconds:

| Piece | What it is |
|---|---|
| `policy/null_server.py` | Websocket server returning all-zero actions. No model, no GPU. Also a real baseline — a null policy scores ~0.093, because Q is scored on the final state and many tasks start with predicates already satisfied. |
| `tests/mock_evaluator.py` | Speaks the real evaluator's CLI and websocket protocol and writes rollout JSONs in the real schema. No simulator. |

`harness/launch.py --eval-module` swaps the mock in for the real evaluator, so the whole
chain runs unchanged:

```bash
# terminal 1 — one null server per eval worker
python -m policy.null_server --base-port 8000 --num-servers 3

# terminal 2 — the full chain, ~2 seconds
python -m harness.launch --config configs/experiments/900-mock-smoke.yaml \
    --eval-module tests.mock_evaluator --workers 3 --base-port 8000 \
    --extra-eval-arg=--fast --output-dir rollouts/mockA
python -m analysis.parse    rollouts/mockA --universe 100 --per-task
python -m analysis.failures rollouts/mockA --sample 2
python -m submission.build  --rollouts rollouts/mockA \
    --wrapper tests/fixtures/mock_wrapper.py \
    --robot-config tests/fixtures/mock_robot.yaml

pytest tests/test_pipeline.py -q
```

**This proves plumbing and schema, never behaviour.** The Q numbers it produces are
synthetic. A green run says the pipeline is wired correctly, not that the policy is good.

The protocol is implemented against the published docs and is **not yet verified against
a BEHAVIOR-1K checkout** — see the assumption list at the top of `policy/wire.py` and
confirm all of it on the first real run.

## Training

The evaluation side of this repo is done. The training side is the critical path.
π₀.₅ trains in the **openpi fork** (`wensi-ai/openpi`, branch `behavior`), which
is cloned as a sibling directory and patched — see [training/README.md](training/README.md)
for what the fork actually contains, quoted from the code.

```bash
git clone -b behavior https://github.com/wensi-ai/openpi.git ../openpi
scripts/apply_openpi_patches.sh              # verifies before it writes
OPENPI_ROOT=../openpi pytest tests/test_progress_head.py -q
OPENPI_ROOT=../openpi python scripts/estimate_memory.py
bash scripts/train_cloud.sh --dry-run        # print the plan, run nothing
```

**A full fine-tune does not fit.** openpi's own README puts it above 70 GB, and
`pi05_b1k` is a full fine-tune with EMA still on. Three configs that do fit:

| Config | Trainable | State | Note |
|---|---|---|---|
| `pi05_b1k` (stock) | 3.353B | 75.0 G | Full fine-tune. Does not fit. |
| `pi05_b1k_frozen_vlm` | 0.430B | 13.5 G | **Preferred.** No new code path. |
| `pi05_b1k_lora` | 0.052B | 7.2 G | Cheaper, but untested for pi05 upstream. |
| `pi05_b1k_frozen_vlm_progress` | 0.430B | 13.5 G | Our contribution. A clean A/B against the second. |

Numbers from `scripts/estimate_memory.py`, which sizes the shipped configs from
the real model graph rather than repeating a table that can drift. They cover
parameters, optimizer moments, EMA and gradients — not activations, which scale
with batch size. Expect **Option A** to be the one that runs: LoRA is cheaper on
paper but is the untested path, and 13.5 G leaves ~27 G of activation headroom
on a 40 GB card.

Two traps that cost a run each, both now pinned by tests:

- **`PathRegex(".*llm.*")` is not "the VLM".** Both experts live inside
  `PaliGemma/llm`, so it freezes the action expert and leaves the 415M SigLIP
  tower trainable — the opposite of the intent, at the same memory cost. Use
  `pi0_config.freeze_vlm_filter()`.
- **A new head aborts checkpoint loading.** `CheckpointWeightLoader` validates
  structural equality after merging, so a parameter absent from `pi05_base` and
  not matched by `missing_regex` fails at startup — after the ~7 GB download.
- **The LoRA filter has the same hole.** `get_freeze_filter` only covers
  `.*llm.*`, so stock LoRA trains all 414M SigLIP parameters — 0.467B trainable,
  *more* than full-expert fine-tuning. Use `pi0_config.lora_freeze_filter()`.

openpi has **no gradient accumulation**. A smaller batch is a genuinely smaller
effective batch, not just a slower step.

## Why the harness matters more than it looks

A full submission is **100 tasks × 10 instances × 1 rollout = 1,000 rollouts**. At the
organizers' published throughput (~13.5 FPS for full-res RGB+depth) plus 150–300s scene
load per trial, one rollout is roughly **20–25 minutes**.

**1,000 rollouts ≈ 350–420 GPU-hours ≈ over two weeks on a single card.**

So evaluation must be fanned out and resumable, or there is no submission. Two configs:

| Config | Scope | Cost | When |
|---|---|---|---|
| **Dev loop** | ~12 tasks × 3 instances = 36 rollouts, instances **10–19**, frozen | ~13 GPU-hr, overnight | Every iteration |
| **Full eval** | 100 × 10 = 1,000 rollouts, instances **0–9** | 350–420 GPU-hr | Only when submitting |

Keep the dev subset frozen so numbers stay comparable, and never tune against 0–9 —
those are what you report.

## Quickstart

```bash
# 1. Provision a cloud GPU box (your laptops don't meet the 32GB RAM / RTX 2070 minimum)
bash scripts/setup_cloud.sh
conda activate behavior

# 2. Pull a 10-task slice (~330 GB). The full LeRobot set is 3.27 TB.
bash scripts/download_data.sh

# 3. Start a policy server (vendor script for the baseline), then get THE NUMBER
bash scripts/first_rollout.sh

# 4. Fan out a sweep
python -m harness.launch --config configs/experiments/001-dev-loop.yaml \
    --workers 4 --base-port 8000 --gpus 0,1,2,3

# 5. Read the results
python -m analysis.parse rollouts/001-dev-loop --universe 100 --per-task
python -m analysis.failures rollouts/001-dev-loop --sample 2

# 6. Package (do this in Week 1 against a throwaway model, not week 7)
python -m submission.build --rollouts rollouts/final \
    --wrapper policy/wrapper.py --robot-config configs/robot/r1pro.yaml
```

Analysis tooling needs `pip install -r requirements-tools.txt` — **not** into the
`behavior` conda env.

## Gotchas that will cost you a day each

- **The pinned tag moves.** It was `v3.9.1`, now `v3.9.2`. Check the
  [evaluation page](https://behavior.stanford.edu/challenge/evaluation.html) before every
  fresh clone and again before submitting.
- **Baseline checkpoints exist for one task only** (`turning_on_radio`). There is no
  pretrained 100-task baseline to run — a real baseline number means training one.
- **First OmniGibson import takes ~5 minutes.** One-time shader compile, not a hang.
- **Hang at `HydraEngine rtx failed creating scene renderer`** → `export OMNIGIBSON_GPU_ID=0`.
- **CuRobo often fails to build.** Install without `--primitives` first.
- **PyPI packages and Docker install are unavailable** during their monorepo migration.
- **Each parallel eval worker needs its own policy server port.** Worker `i` uses
  `base_port + i`. The IP-submission mode requires ≥50 ports for exactly this reason.

## Scoring, and what it implies

Q = fraction of BDDL goal predicates satisfied **at episode end**, averaged over all 100
tasks. Two consequences worth internalising:

1. **"At episode end" is literal.** Complete the task, then knock the object back out
   while closing a door, and you score zero. Stability matters as much as achievement —
   which is why `early_stop_on_goal` is in the experiment config.
2. **Broad shallow competence beats narrow depth.** 20 tasks at Q=0.5 with 80 zeros gives
   mean 0.10. Every task at Q=0.15 gives 0.15. Reliably completing the *first* predicate
   of all 100 tasks outscores perfectly solving fifteen.

## Comparing two runs: always pair

The evaluator is nondeterministic and we run one rollout per instance. Comparing two
*independent* 36-rollout sweeps needs roughly **ΔQ > 0.115** to clear the noise — wider
than the gap between 1st and 5th place in 2025. An unpaired A/B at dev-loop size cannot
resolve the differences we care about.

Run both arms on the **same instances** and difference per instance instead. The
instance-to-instance variance — most of the total, because some instances are simply
harder — cancels, and the detectable difference drops to about **0.033**. Same GPU spend,
~3.5x the resolution.

```bash
python -m analysis.compare rollouts/baseline rollouts/candidate --per-instance
```

It reports mean ΔQ, standard error, a 95% CI on the paired difference, and what the same
rollouts would have resolved unpaired. If the two runs do not cover identical
`(task, instance_id, rollout_id)` keys it warns loudly and refuses to report — a broken
pairing throws away the entire advantage. This is why the dev subset stays frozen.

## Targets

| Tier | Mean Q | Meaning |
|---|---|---|
| Floor | > 0.000 | A valid submission. Four teams failed to clear this in 2025. |
| Target | ≥ 0.10 | Would have placed 4th–5th in 2025. |
| Stretch | — | The $1,000 Outstanding Open Source prize. Of 2025's top five, only two released code. |

2025 winner: Q = 0.260, 12.4% success rate.

## Status

| Component | State |
|---|---|
| `harness/launch.py` | Working. Verify the evaluator CLI against your checkout on first run. |
| `analysis/parse.py` | Working. Field names match the published JSON schema. |
| `analysis/failures.py` | Working, thresholds uncalibrated — tune after watching real rollouts. |
| `submission/build.py` | Working. |
| `policy/server.py` | **Skeleton.** Protocol documented, API not yet verified. Use vendor serve scripts until then. |
| `policy/null_server.py` | Working standalone. Zero-action baseline + pipeline exerciser. Protocol unverified against a real checkout. |
| `policy/wire.py` | Working. Single source of truth for the wire format; lists every unverified assumption. |
| `tests/mock_evaluator.py` | Working. Same CLI + protocol + output schema as the real evaluator, no simulator. |
| `analysis/compare.py` | Working. Paired A/B with CI and a coverage guard. |
| `tests/test_pipeline.py` | 16 tests, all passing. |
| `training/patches/` | Working. Applied to the openpi fork by `scripts/apply_openpi_patches.sh`. |
| `scripts/estimate_memory.py` | Working. Sizes any openpi TrainConfig from the real model graph. |
| `scripts/train_cloud.sh` | Written, `--dry-run` verified. **Never run end to end** — no GPU here. |
| `tests/test_progress_head.py` | 14 tests, all passing on CPU. |
| progress-head labels | **Missing.** The head is wired and gradients reach it; nothing produces the label yet. |
| `docker/policy-server/` | Skeleton. |
| `policy/wrapper.py` | **Missing.** Required for submission; `tests/fixtures/mock_wrapper.py` is a test stand-in, not a substitute. |
| `configs/robot/r1pro.yaml` | **Missing.** Copy it from the BEHAVIOR-1K checkout; see `configs/robot/README.md`. |

## License

MIT — see [LICENSE](LICENSE). The challenge's $1,000 Outstanding Open Source prize goes
to the top-performing open-source submission, so this is an eligibility requirement, not
a formality.

## Links

- [Evaluation & rules](https://behavior.stanford.edu/challenge/evaluation.html) ·
  [Submission](https://behavior.stanford.edu/challenge/submission.html) ·
  [Baselines](https://behavior.stanford.edu/challenge/baselines.html) ·
  [Dataset](https://behavior.stanford.edu/challenge/dataset.html)
- [Discord](https://discord.gg/bccR5vGFEx) — office hours Mondays 5–6pm Pacific
- 2025 winner: [write-up](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html) ·
  [paper](https://arxiv.org/abs/2512.06951) · [code](https://github.com/IliaLarchenko/behavior-1k-solution)
