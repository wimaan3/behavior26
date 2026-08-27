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
harness/    parallel rollout orchestration — the thing that makes iteration possible
analysis/   rollout JSON -> dataframe -> failure taxonomy
submission/ package + validate the final zip
configs/    one committed YAML per experiment; robot config
scripts/    cloud setup, data download, first-rollout timing
docker/     policy server image (a required submission artifact)
docs/       project charter
```

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
| `docker/policy-server/` | Skeleton. |

## Links

- [Evaluation & rules](https://behavior.stanford.edu/challenge/evaluation.html) ·
  [Submission](https://behavior.stanford.edu/challenge/submission.html) ·
  [Baselines](https://behavior.stanford.edu/challenge/baselines.html) ·
  [Dataset](https://behavior.stanford.edu/challenge/dataset.html)
- [Discord](https://discord.gg/bccR5vGFEx) — office hours Mondays 5–6pm Pacific
- 2025 winner: [write-up](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html) ·
  [paper](https://arxiv.org/abs/2512.06951) · [code](https://github.com/IliaLarchenko/behavior-1k-solution)
