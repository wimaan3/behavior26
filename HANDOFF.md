# HANDOFF — training

Work is on `main` and pushed.

**Canonical checkout: `/mnt/c/Users/Imaan Soltanalipour/robotProj/behavior26`.**
Git is slower there, but it is the copy with the sibling `openpi` and
`external/b1k` checkouts, so it runs strictly more of the suite than a bare
clone does — that is what makes it the one worth trusting. Work there.

---

## 2026-09-11 session: Jetson relabel merged, pairing bug fixed, power re-run

**Heads-up for whoever picks this up: the local working tree at
`/home/imaansol/3MagicLabs/stanbeh` was EMPTY at the start of this session** —
no `.git`, no files, and no clone of `wimaan3/behavior26` anywhere on the
machine. Everything below was done against a fresh clone of `origin`. Nothing
was lost that had been pushed, but any unpushed local work from before
2026-09-05 is gone. Push early.

### Done

| # | Task | Result |
|---|---|---|
| 1 | Merge `origin/jetson/labels` into `main` | `7322af4`. Clean, **no conflicts** — `.gitattributes` was already identical on both sides (the Jetson's eol=lf pins won in the earlier merge `076fe29`), so the overlap from last time did not recur. Brings `a6a5a4a` + `d3b1769`: progress re-anchored at episode start, new `graded` tier, new schema columns. |
| 2 | `compare.py` pairing bug | `66d365d`. Was joining on `rollout_id`. Now pairs on (task, instance) with seeds averaged within each arm. See below. |
| 3 | Re-run `power.py` on the new shortlist | `eee5b9d`. Binary noise model replaced with a graded one; tables regenerated into a dated revision of `docs/AB_PROTOCOL.md`. |
| 4 | Budget / partial mode | Already implemented on `main` before this session (`de7c131`) — verified working end-to-end, not rewritten. Stale on-demand figures in `WEEK1_CHARTER.md` superseded. |
| 5 | Fill in `001-dev-loop.yaml` | `d7e64d4`. Tasks were already correct; added the reserve, regenerated the figures the relabel invalidated, fixed two wrong claims, added tests. |

### The pairing bug, in one line

`JOIN_KEYS` included `rollout_id`, so arm A's rollout 3 was paired with arm B's
rollout 3 — unrelated draws under a nondeterministic simulator. It also counted
`n*m` pairs where there are only `n` units, shrinking the SE by ~sqrt(m). On a
6-instance x 3-seed case with a real, near-constant +0.020 effect, the old key
reported "not significant" with a detectable floor of 0.25; the fix recovers the
effect at a floor of 0.0015. **Latent at `num_rollouts: 1`** — the dev loop's
existing numbers are unaffected, and a test pins that.

### What changed about the A/B design

`sigma_w` on the two tasks we run is **2.7x lower** than the binary model
assumed, and the MDE at N=40 goes 0.145 -> 0.058. Consequences, all recorded in
the 2026-09-11 revision of `docs/AB_PROTOCOL.md`:

- Instances beat seeds at equal cost. **1 seed is the answer** across the whole
  plausible noise range.
- §2's noise-floor measurement no longer decides the design — it only calibrates
  how much to believe the MDE. It no longer blocks the first A/B.
- **`n=3` is the binding constraint.** The frozen instance list `[10, 11, 12]`
  gives N=6 and an MDE of 0.181 — coarser than the effect the pairing exists to
  resolve. n=20 reaches 0.058 for 22.3 GPU-hr / $11.

### Open / next

1. ~~**Decide on `n`.**~~ **DONE** — revisions 2026-09-11b and 2026-09-11c took
   the frozen list from `[10, 11, 12]` to **n=27, instances 10–36**. MDE
   0.181 → **0.049** at f=0.20 for $9–15.

2. **Know the floor before reading any result.** This design resolves
   **ΔQ ≥ ~0.05 on its two tasks**. The binding constraint is **k=2 tasks, not
   n**: the task-level variance term is σ_task²/k, which no number of instances
   or seeds touches. At k=2 it is one contrast on one degree of freedom, so a
   claim "about BEHAVIOR" is not available at any budget we can reach. If that
   claim is what's wanted, the next spend is **more tasks** (k=4 at n=27), not
   more instances. Full arithmetic in revision 2026-09-11c.

3. **The inconclusive-result response is pre-registered** (revision 2026-09-11c
   §3) — manipulation check on `progress_loss` first, then one of {larger λ /
   more instances / different task pair} by where the point estimate falls and
   whether the two tasks agree. Decided before any number exists. Do not
   re-litigate after seeing the result.
4. Gaps 2–7 in `docs/AB_PROTOCOL.md` §4 remain (gap 1 is fixed, gap 5 is
   defused for the graded tier only). Gaps 2 and 3 — median pass Q and its
   spread — are what §3.3 requires for the headline and are still unwritten.
5. `sigma_b = 0.05` is still assumed, never estimated. It is now ~16% of
   Var(d_i) at f=0.20 rather than ~2%, so it matters more than it used to.
6. The Jetson's reward tests (`analysis/reward/tests/`, 54 of them) **all skip
   here** — they gate on a LeRobot pull at `/home/imaansol/behavior-data`, which
   does not exist on this box. The relabel logic is unverified locally; it was
   tested on the Jetson. `tests/test_progress_labels.py` (the merge-script
   schema pin) does run and passes.

## 2026-09-04 session: reading the real evaluator

`external/b1k/OmniGibson` (v3.9.2) and `../openpi` are now checked out locally.
Reading them instead of the docs contradicted several things this repo believed.
**Where this file and the code disagree, the code wins.**

### Corrections to previously-stated facts

| We said | The source says |
|---|---|
| Full submission = 100 × 10 = **1,000** rollouts, ~350–420 GPU-hr | `TEST_INSTANCE_IDS = range(301, 341)`, `NUM_PUBLIC_TEST_INSTANCES = 20` → **2,000** rollouts, ~700–840 GPU-hr |
| Leaderboard instances are **0–9**, dev set is 10–19 | Instance ids are **301–340**; public split is 301–320. `--instance-indices` are *indices into the split*, not ids. There is no free holdout inside `public_test` — use `--mode train` for the dev loop. |
| Rollout filenames were a guess (`{task}_inst{i}_rollout{r}.json`) | `f"{task}_{instance}_{rollout}.json"` (eval.py l.184), and `compute_final_q_score` **asserts** on every `*.json` in `json/` |
| Wire codec is msgpack-numpy | It is BEHAVIOR-1K's own `__ndarray__` msgpack extension. network_utils.py says explicitly why not msgpack-numpy. **Incompatible formats.** |
| Metadata frame is optional | Mandatory: the client blocks on `unpackb(conn.recv())` before sending anything |
| `policy/wrapper.py` is a missing required artifact | `omnigibson.eval.wrappers.{DefaultWrapper,RGBDFullResWrapper}` already ship; a custom wrapper is optional |
| `configs/robot/r1pro.yaml` is missing | Copied verbatim from the checkout; now present |

### Bugs fixed this session (each would have failed or wasted a paid run)

1. **Wire codec.** `policy/wire.py` packed with msgpack-numpy. The evaluator's
   `unpack_data` looks for `b"__ndarray__"`, so an msgpack-numpy array decodes
   as a plain dict and `th.from_numpy(dict)` raises on step 1 of rollout 1.
2. **Null server sent Python lists.** The client does
   `th.from_numpy(deepcopy(action_dict["action"]))`, which needs a real ndarray.
3. **Null server replied to `{"reset": True}`.** The client's `reset()` sends it
   and does *not* read a reply; the real server answers with `continue`.
   Replying leaves an unread frame, so every later `recv()` returns the previous
   step's action — an episode-long one-step lag, silently.
4. **Submission package layout.** The scorer parses the directory name as
   `<track>.<testset>.<team>.<affiliation>.<date>` and asserts on filenames.
5. **`--resume` never matched.** `already_done` compared job indices (0,1,2)
   against the resolved ids on disk (301,302,303), so a half-finished sweep
   re-ran every job — while still printing the count it had found, so it looked
   fine. Never worked against the real evaluator in test mode; the mock hid it
   by writing indices back out. Hundreds of rented GPU-hours on a resumed
   2,000-rollout run.
6. **The dev loop was tuning on the leaderboard.** `001-dev-loop.yaml` said
   "instances 10-19 only", which selects public-test instances 311-320 — all
   scored. Corrected to `mode: train`.

### Scoring consequence worth internalising

`compute_final_q_score` divides by a **fixed** denominator (20 for the public
set), so a partial submission is not scored on what it covers — the missing
instances are averaged in as zeros. Covering 10 of 20 instances halves the score.


The evaluation side is done and documented in the section below. This session
built the training side, which was entirely missing and is the critical path.

## Done this session

| Task | Result |
|---|---|
| 1. Read the fork | [training/README.md](training/README.md) — quoted from the code, not remembered. |
| 2. Configs that fit | `pi05_b1k_frozen_vlm`, `pi05_b1k_lora`, `pi05_b1k_frozen_vlm_progress` + `scripts/estimate_memory.py`. |
| 3. Progress head | Linear readout on `suffix_out`, BCE loss, config knobs for weight and dimensionality. |
| 4. Gradient proof | `tests/test_progress_head.py`, 14 tests, ~80s on CPU. Gradients reach the head. |
| 5. Launch script | `scripts/train_cloud.sh`. `--dry-run` walks the whole plan; **never run end to end**. |

Measured state (`scripts/estimate_memory.py`), parameters + AdamW + EMA +
gradients, before activations:

| config | trainable | state | verdict |
|---|---|---|---|
| `pi05_b1k` (stock, full FT) | 3.353B | 75.0 G | does not fit; reproduces openpi's ">70 GB" |
| `pi05_b1k_frozen_vlm` | 0.430B | 13.5 G | **expected to be the one we run** |
| `pi05_b1k_lora` | 0.052B | 7.2 G | cheaper, untested path |
| `pi05_b1k_frozen_vlm_progress` | 0.430B | 13.5 G | our contribution |

```bash
git clone -b behavior https://github.com/wensi-ai/openpi.git ../openpi
scripts/apply_openpi_patches.sh
OPENPI_ROOT=../openpi pytest tests/test_progress_head.py -q   # 15 passed, ~70s
OPENPI_ROOT=../openpi python scripts/estimate_memory.py
bash scripts/train_cloud.sh --dry-run
```

The two test suites need two environments (`requirements-tools.txt` vs
`requirements-training.txt`); `pytest tests/` in either fails the other on a
missing import. See `pytest.ini`. Verified separately: 16 passed / 15 passed.

## Traps found in the fork (each would have cost a run)

1. **`PathRegex(".*llm.*")` freezes the wrong half.** Both experts live inside
   `PaliGemma/llm`; only the action expert's leaves carry a `_1` suffix. So the
   naive regex freezes the action expert (2.936B) and leaves the SigLIP tower
   (0.415B) trainable — the opposite of the intent, at the same memory cost, so
   nothing looks wrong. Use `pi0_config.freeze_vlm_filter()`.
2. **A new head aborts checkpoint loading.** `_merge_params` returns the
   intersection plus `missing_regex` matches, and `_load_weights_and_validate`
   then asserts structural equality — so a parameter absent from `pi05_base` and
   unmatched raises at startup, after the ~7 GB download. Hence
   `CheckpointWeightLoader.missing_regex`.
3. **`preprocess_observation` silently drops new `Observation` fields.** It
   rebuilds the dataclass field by field instead of using
   `dataclasses.replace`, so a progress label would never reach the model while
   the config claimed the head was enabled.
4. **The LoRA freeze filter has the same hole as (1).** `get_freeze_filter`
   only covers `.*llm.*`, so stock LoRA trains all ~414M SigLIP parameters —
   0.467B trainable and 14.2 G, *more* than full-expert fine-tuning, which is
   not what the ">22.5 GB" row in openpi's README suggests you are buying.
   `lora_freeze_filter` brings it to 0.052B / 7.2 G.
5. **pi05 starts with its adaRMS gates closed.** Zero-init modulation Denses
   mean that at step 0 only six parameter groups get gradient, and `suffix_out`
   does not depend on the images or the prompt at all. Not a bug — but a step-0
   `grad_norm` check on the expert's MLP reads as a broken model.

## Task 7 — 001-dev-loop.yaml: still blocked

`configs/experiments/001-dev-loop.yaml` still has `tasks: []`. It is blocked on
the Jetson's full-scale phi_0 sweep and was **not** filled in this session.

When the sweep lands, pick 12 tasks: mostly short and low-D for score, but
**including 2–3 Type C tasks** even though they will score badly — the file's own
rule 3 explains why. Do not quietly pick only the easy ones.

Known bad, exclude: `make_microwave_popcorn` (collapsed instrumentation, despite
an organizer recommending it), `putting_dishes_away_after_cleaning`,
`rearranging_kitchen_furniture`, `storing_food`.

One thing that changed: the dev loop must run with **`--mode train`**, not
`public_test`. Indices 10–19 of the public split are instances 311–320, which are
scored — there is no free holdout inside `public_test`.

## Pending — training

- **No progress labels exist.** The head is wired, allocated and receives
  gradient; nothing produces the target yet. `progress_key` is `None`, so
  `pi05_b1k_frozen_vlm_progress` currently trains identically to
  `pi05_b1k_frozen_vlm`. This is the next blocking item.
- **`pi05_b1k_lora` is untested for pi05 upstream.** openpi ships LoRA configs
  for pi0 and pi0-FAST only. The machinery is not pi0-specific, so it is
  expected to work, but budget debugging before relying on it.
- **`scripts/train_cloud.sh` has never run end to end.** No GPU here. Run it
  with `--dry-run` on the box first.
- **`dataset_root` in `pi05_b1k` is a Stanford cluster path.** Our configs
  default to `./data/b1k/<task>`; the launch script always passes an explicit
  local path.
- Nothing here measures whether the progress head *helps*. That needs
  `analysis/compare.py` across two runs on identical instances.

## Pending — evaluation (carried over, unchanged)

- **Verify the websocket protocol against a real BEHAVIOR-1K checkout.** See the
  assumption list at the top of `policy/wire.py`.
- **`policy/wrapper.py` does not exist** — required submission artifact.
- **`configs/robot/r1pro.yaml` does not exist** — required submission artifact.
  `tests/fixtures/` holds test stand-ins. Do not submit them.
- `configs/experiments/001-dev-loop.yaml` still has `tasks: []`.
- LICENSE copyright holder name still unconfirmed.
- `analysis/census/*` has uncommitted CRLF-only churn, untouched by either
  session. Consider a `.gitattributes` with `* text=auto`.

## Previous session — GPU-free test harness (`local/test-harness`, 7 commits)

Full chain runs in ~2s with no GPU:
`null_server → mock_evaluator → harness/launch.py → parse → failures → compare → build`.
16 tests passing. Four bugs found and fixed in pre-existing code:

1. `harness/launch.py` `already_done()` — substring filename matching reported
   never-run instances as complete. **Silent zero-scoring on a resumed sweep.**
2. `analysis/failures.py` — `CRASHED` unreachable (parse defaulted missing q to 0.0).
3. `analysis/failures.py` — `TIMEOUT` unreachable (`timeout_ratio` never referenced).
4. `analysis/failures.py` — `x or 0.0` does not default NaN (NaN is truthy).

All four funnelled bad rollouts into `ACTIVE_NO_PROGRESS`.
