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

### A green local suite is NOT evidence the box will work (2026-09-12)

**Our tests validate LOGIC. They do not validate the ENVIRONMENT.** Four defects
in one afternoon on the pod, all of which passed locally, three of which passed
*green* rather than failing:

| # | Defect | Why local was green |
|---|---|---|
| 1 | `pyarrow` missing from `requirements-tools.txt` | Dev machine had pyarrow from elsewhere. On the pod, `test_progress_labels.py` module-level `importorskip` meant it never COLLECTED — 20 tests silently absent. Pod ran 98, dev ran 121. Both "passed". |
| 2 | RunPod image ships no conda | Dev machine has `/home/imaansol/miniconda3/bin`, so the conda branch never executed locally. `setup.sh` died at 2 min with `ERROR: Conda not found`. |
| 3 | A bare clone has no sibling `openpi` / `external/b1k` | Those tests skip, including the live openpi↔evaluator robot-name check — the one that proves the first rollout will not KeyError. |
| 4 | `setup_cloud.sh` tests **prepended** the stub dir to the real PATH | The developer's own `conda`/`git`/`nvidia-smi` shadowed the stubs, so the tests exercised a different branch than production. This is what let #2 survive being "covered by tests". |

#### Why this class hides so well

- **`importorskip` at module level does not fail — it deletes.** The module
  drops out of collection and its tests disappear from the run. The summary
  still says "all passed", just with a smaller number nobody compares.
- **Prepending to PATH imports the host.** A test that stubs a binary but leaves
  the real PATH behind is testing the developer's machine, not a bare box.
- **Presence of a sibling directory changes the suite.** The robotProj checkout
  runs strictly more tests than a bare clone; see
  [[behavior26-canonical-checkout]].
- **The box lies about itself.** `nproc` reports 192 while the cgroup quota is
  20.4 CPUs; `free` reports 723 GB while the limit is 84 GB. Anything sizing a
  worker pool from `nproc` will thrash.

#### Countermeasures now in place

- `scripts/preflight.sh` asserts **every test module collects ≥1 test**, so a
  vanished module fails loudly instead of shrinking the count. Verified by
  planting a module that cannot import: pytest said "90 passed", preflight
  failed it.
- `tests/test_setup_cloud.py` **replaces** PATH (stub + `/usr/bin` + `/bin`)
  rather than prepending, and stubs `conda` so a unit test cannot reach the
  network.
- Preflight runs **on the target box**, before anything expensive.

#### Before session B — the surface is much larger and failures cost more

Session A's environment surface was small and a failure cost $0.14. Session B
adds CUDA/driver compatibility, jax and torch wheels, the openpi fork and its
deps, a 330 GB dataset, and VRAM limits — and a failure there wastes a training
run, not two minutes. **Run preflight on that box first**, and extend it to
assert the things session B actually depends on rather than assuming them:

- `openpi` importable, and the **live** robot-name check passing — not skipped;
- jax sees the GPU, and the CUDA/driver pair is one openpi supports;
- the cgroup CPU and memory limits (not `nproc`/`free`), and VRAM headroom
  against the 3.3B model;
- `progress_loss` present in the loss dict — the manipulation check in
  AB_PROTOCOL revision 2026-09-11c §3 depends on it, AND every λ ratio is
  meaningless without it (absent key means `loss == action_loss`, i.e. the head
  is enabled but unsupervised);
- **`action_loss` and `progress_loss` magnitudes at step 0 and over the first
  100 steps** — this calibrates λ. `progress_loss_weight = 0.1` was set before
  anyone had seen either loss's scale, and it is the one hyperparameter we
  cannot sweep. Since `loss = A + λ·P` with both terms comparable, λ is solvable
  rather than guessable: `λ = share·A/((1−share)·P)`, target share 10–30%. If
  `P` is ~100× smaller than `A`, λ=0.1 makes the head nearly inert and the A/B
  returns a null for a trivial reason. Report the trend across the 100 steps,
  not just step 0 — `A/P` drifts as both terms learn. Full rule in §1a;
- dataset root on **local disk**, not the network volume (`train_cloud.sh:132`
  warns data loading will dominate otherwise).

Treat "it passed on my machine" as saying nothing about any of these.

### Session A checklist additions (2026-09-12)

- Time `import omnigibson` on the volume env — **the 2nd and 3rd runs, not the
  1st**. The first includes a one-time shader compile (~5 min) and, because
  `OMNIGIBSON_APPDATA_PATH` is on container disk by design, every new pod pays
  it again. The 2nd/3rd are the steady-state per-process cost.
- Context for reading it: the import is paid **per job, not per rollout**
  (`build_jobs` amortizes it; `--instances-per-job` defaults to 0 = one job per
  task), so a k=2 cycle pays 4 imports. It only becomes material under
  parallelism. Full reasoning and the fallbacks are in AB_PROTOCOL §1a.

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

3. **OPEN DECISION before shot one: k=2 vs k=4.** Revision 2026-09-11d. This is
   the largest single gain in claim strength available — generalisable MDE
   **0.199 → 0.042** for **+38.2 GPU-hr ≈ +$11–19** of eval — and it turns
   "helps on these two tasks" into "helps on BEHAVIOR". It hinges on whether
   training cost tracks **steps** or **tasks**: π₀.₅ trains multi-task, so four
   tasks at the same step budget may be the same gradient steps on more diverse
   data. **Session B must report** steps/sec for a 2- vs 4-task mix, loss-vs-step
   for both, and peak memory. Decision rule is pre-committed in 2026-09-11d.
   3rd task `outfit_a_basic_toolbox`; 4th `preparing_lunch_box` — *not*
   `thawing_frozen_food`, which our own defect report flags (18/200 demos reach
   the goal). Both blockers are now cleared ahead of the edit: the grading
   threshold is restated to 0.5 on the concept, and the reserve is to be left
   deliberately empty.

4. **ARM SYMMETRY is now a hard rule (§3.6), same status as same-commit.** The
   label pipeline cannot label every episode (5 of 200 for the shoe rack, 1 for
   the coffee station). Both arms must train on the SAME `--drop-unlabelled`
   filtered root; arm A just does not map the progress column. Training the
   baseline on the pristine root — which is what `merge_progress_labels.py`
   used to advertise as a virtue — would leave the arms differing by ~2.5% of
   the data as well as by the head. The merge script now has the drop mode and
   writes `meta/progress_filter.json` so the claim is auditable afterwards.

5. **The inconclusive-result response is pre-registered** (revision 2026-09-11c
   §3) — manipulation check on `progress_loss` first, then one of {larger λ /
   more instances / different task pair} by where the point estimate falls and
   whether the two tasks agree. Decided before any number exists. Do not
   re-litigate after seeing the result.
6. Gaps 2–7 in `docs/AB_PROTOCOL.md` §4 remain (gap 1 is fixed, gap 5 is
   defused for the graded tier only). Gaps 2 and 3 — median pass Q and its
   spread — are what §3.3 requires for the headline and are still unwritten.
7. `sigma_b = 0.05` is still assumed, never estimated. It is now ~16% of
   Var(d_i) at f=0.20 rather than ~2%, so it matters more than it used to.
8. The Jetson's reward tests (`analysis/reward/tests/`, 54 of them) **all skip
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
