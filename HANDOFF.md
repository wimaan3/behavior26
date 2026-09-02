# HANDOFF — training

Branch `local/training` (from `local/test-harness`), **not pushed**.

The evaluation side is done and documented in the section below. This session
built the training side, which was entirely missing and is the critical path.

## Done this session

| Task | Result |
|---|---|
| 1. Read the fork | [training/README.md](training/README.md) — quoted from the code, not remembered. |
| 2. Configs that fit | `pi05_b1k_frozen_vlm`, `pi05_b1k_lora`, `pi05_b1k_frozen_vlm_progress` + `scripts/estimate_memory.py`. |
| 3. Progress head | Linear readout on `suffix_out`, BCE loss, config knobs for weight and dimensionality. |
| 4. Gradient proof | `tests/test_progress_head.py`, 14 tests, ~80s on CPU. Gradients reach the head. |
| 5. Launch script | `scripts/train_cloud.sh`. `--dry-run` verified; **never run end to end**. |

```bash
git clone -b behavior https://github.com/wensi-ai/openpi.git ../openpi
scripts/apply_openpi_patches.sh
OPENPI_ROOT=../openpi pytest tests/test_progress_head.py -q
OPENPI_ROOT=../openpi python scripts/estimate_memory.py
```

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
4. **pi05 starts with its adaRMS gates closed.** Zero-init modulation Denses
   mean that at step 0 only six parameter groups get gradient, and `suffix_out`
   does not depend on the images or the prompt at all. Not a bug — but a step-0
   `grad_norm` check on the expert's MLP reads as a broken model.

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
