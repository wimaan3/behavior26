# behavior26

A **predicate-aware VLA** for the [2026 Stanford BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html).
The contribution is an auxiliary **progress head** on π₀.₅: a second output that
predicts how much of the task's BDDL goal is already satisfied, trained
alongside the action decoder so the shared representation has to encode task
progress rather than just the next action.

**Deadline: 16 October 2026.** Budget: $200 total, all rented GPU.

---

## Status

**Nothing has run on a GPU yet.** Every number below came from source-reading,
a CPU, or the mock evaluator. Treat all of it as unverified against hardware.

| | |
|---|---|
| **Works, exercised end to end** | Rollout harness (parallel, resumable), mock evaluator, rollout parsing, failure taxonomy, paired A/B, submission packaging. All on a laptop, no GPU. |
| **Works, CPU-verified only** | The progress head: builds, produces finite loss, and gradients reach it and the shared trunk. openpi patches apply cleanly; three training configs fit in 32 GB by measurement. |
| **Untested — needs a GPU** | Every training config. `scripts/train_cloud.sh` (dry-run only). Serving a real checkpoint. LoRA on π₀.₅ is untested upstream too. |
| **Not built** | Progress labels are not attached to any dataset — the merge script and the Jetson's label pipeline exist and their schemas match, but they have never been run together. `policy/server.py` has no model loading. |
| **Known cap** | We train 2 tasks, so the submission ceiling is **Q = 0.020** (Q averages over all 100 tasks; 98 score zero). Accepted deliberately — see [AB_PROTOCOL](docs/AB_PROTOCOL.md). |

---

## How to use it

### 1. Run the whole pipeline with no GPU (~2 minutes)

```bash
git clone <this repo> && cd behavior26
pip install -r requirements-tools.txt

# terminal 1 — one null policy server per eval worker
python -m policy.null_server --base-port 8000 --num-servers 3

# terminal 2 — the full chain against a mock evaluator
python -m harness.launch --config configs/experiments/900-mock-smoke.yaml \
    --eval-module tests.mock_evaluator --workers 3 --base-port 8000 \
    --extra-eval-arg=--fast --output-dir rollouts/mockA
python -m analysis.parse    rollouts/mockA --universe 100 --per-task
python -m analysis.failures rollouts/mockA --sample 2
python -m submission.build  --rollouts rollouts/mockA \
    --wrapper tests/fixtures/mock_wrapper.py \
    --robot-config configs/robot/r1pro.yaml \
    --team behavior26 --affiliation 3MagicLabs \
    --intended-tasks-from configs/experiments/001-dev-loop.yaml

pytest tests/test_pipeline.py tests/test_progress_labels.py -q
```

This proves plumbing and schema, never behaviour. The Q numbers are synthetic.
Details: [docs/GPU_FREE_TESTING.md](docs/GPU_FREE_TESTING.md).

### 2. Check the training side, still no GPU

```bash
git clone -b behavior https://github.com/wensi-ai/openpi.git ../openpi
scripts/apply_openpi_patches.sh                       # verifies before writing
OPENPI_ROOT=../openpi pytest tests/test_progress_head.py \
    tests/test_shipped_configs.py tests/test_robot_config.py -q
OPENPI_ROOT=../openpi python scripts/estimate_memory.py
python analysis/power.py                              # A/B design + cost
bash scripts/train_cloud.sh --dry-run                 # walks the whole plan
```

### 3. A real rollout (needs a rented GPU)

```bash
# The conda env and the 29.3 GB dataset go on the NETWORK VOLUME, not container
# disk — so a second pod mounting the same volume starts with the stack ready.
# Confirm the mount point against the console; setup_cloud.sh refuses to run if
# it is container disk wearing the right name.
VOLUME_ROOT=/workspace bash scripts/setup_cloud.sh   # pinned tag — CHECK IT FIRST
source /workspace/env.sh             # sets CONDA_ENVS_PATH, then activates
bash scripts/download_data.sh        # a slice; the full set is 3.27 TB
bash scripts/first_rollout.sh        # get THE NUMBER
```

`source env.sh`, not `conda activate behavior` — the env lives at a prefix on
the volume and conda only finds it by name via `CONDA_ENVS_PATH`. A conda env is
not path-relocatable, so every pod must mount the volume at the **same** path.

Then train and evaluate: [docs/TRAINING.md](docs/TRAINING.md),
[docs/AB_PROTOCOL.md](docs/AB_PROTOCOL.md).

---

## Three traps that cost a run each

All three are fixed here and pinned by tests; they are listed because they are
upstream defects that will bite anyone else who builds on the same code.

1. **The wire codec is not msgpack-numpy.** BEHAVIOR-1K ships its own
   `__ndarray__` msgpack extension. An msgpack-numpy array decodes as a plain
   dict and `th.from_numpy` raises on the first step. The action must also be a
   real `ndarray`, and a `{"reset": True}` frame must not be answered.
2. **`PathRegex(".*llm.*")` freezes the wrong half.** Both experts live inside
   `PaliGemma/llm`, so it freezes the action expert and leaves the 415M SigLIP
   tower trainable — same memory cost, opposite intent. Use
   `pi0_config.freeze_vlm_filter()`. openpi's own LoRA filter has the same hole.
3. **The robot name disagrees between two files.** `r1pro.yaml` says
   `robot_r1`, openpi's `b1k.py` said `robot`, and every observation key is
   prefixed with it — `KeyError` on step 1.

---

## Instance ids: 301–340, not 0–19

Verified against BEHAVIOR-1K v3.9.2 (`omnigibson/eval/utils/eval_utils.py`):

```python
TEST_INSTANCE_IDS = list(range(301, 341))   # public 301-320, hidden 321-340
NUM_PUBLIC_TEST_INSTANCES = 20
```

Anything below 301 is a **training** instance. `--instance-indices` are
*indices into a split*, not ids — index 0 of `public_test` is instance 301.
Configs here declare real ids and the harness converts.

Two rules follow, both enforced in `harness/launch.py`:

- **The dev loop uses training instances.** All 20 public instances are scored,
  so there is no holdout inside them.
- **Missing rollouts are zeros, not omissions.** `compute_final_q_score`
  divides by a fixed 20 per task.

Full detail: [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Layout

```
policy/      model + websocket server (what the evaluator talks to)
harness/     parallel, resumable rollout orchestration
analysis/    rollout JSON -> dataframe -> failure taxonomy -> paired A/B
             reward/ : the Jetson's per-frame progress labels
training/    openpi fork patches + the progress head
submission/  package + validate the final zip
tests/       mock evaluator + the GPU-free suites
configs/     one committed YAML per experiment; robot config
docs/        protocol, evaluation, training, gotchas
```

| Doc | |
|---|---|
| [AB_PROTOCOL.md](docs/AB_PROTOCOL.md) | **Frozen.** A/B design, power, cost, stopping rule |
| [EVALUATION.md](docs/EVALUATION.md) | Instance ids, scoring, why to pair |
| [TRAINING.md](docs/TRAINING.md) | Configs that fit, memory measurements |
| [training/README.md](training/README.md) | What the openpi fork contains, quoted from source |
| [GPU_FREE_TESTING.md](docs/GPU_FREE_TESTING.md) | The no-GPU path in full |
| [GOTCHAS.md](docs/GOTCHAS.md) | Setup traps |
| [HANDOFF.md](HANDOFF.md) | Current session state |

---

## Targets

| Tier | Mean Q | |
|---|---|---|
| Floor | > 0.000 | A valid submission. Four teams failed this in 2025. |
| Our cap | 0.020 | 2 tasks trained; 98 score zero. Budget-bound. |
| 2025 winner | 0.260 | 12.4% success rate |

The leaderboard position is not the goal at this budget. The deliverable is a
clean, reproducible A/B on whether a progress head helps, plus infrastructure
that is worth publishing — of 2025's top five, only two released code.

---

## License

MIT — see [LICENSE](LICENSE). Required for the $1,000 Outstanding Open Source
prize.

## Links

[Evaluation & rules](https://behavior.stanford.edu/challenge/evaluation.html) ·
[Submission](https://behavior.stanford.edu/challenge/submission.html) ·
[Baselines](https://behavior.stanford.edu/challenge/baselines.html) ·
[Dataset](https://behavior.stanford.edu/challenge/dataset.html) ·
[Discord](https://discord.gg/bccR5vGFEx) ·
2025 winner [write-up](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html) ·
[code](https://github.com/IliaLarchenko/behavior-1k-solution)
