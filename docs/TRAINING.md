# Training

Split out of the README. See [../training/README.md](../training/README.md)
for what the openpi fork actually contains, quoted from its source.

---


The evaluation side of this repo is done. The training side is the critical path.
π₀.₅ trains in the **openpi fork** (`wensi-ai/openpi`, branch `behavior`), which
is cloned as a sibling directory and patched — see [../training/README.md](../training/README.md)
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
