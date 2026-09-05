# Training

Everything here targets the **openpi fork**, `wensi-ai/openpi` branch `behavior`,
which is where π₀.₅ actually trains on BEHAVIOR data. The fork is not vendored:
it is cloned as a sibling directory and our changes are applied as patches.

```bash
git clone -b behavior https://github.com/wensi-ai/openpi.git ../openpi
scripts/apply_openpi_patches.sh --check     # verify before writing
scripts/apply_openpi_patches.sh             # apply
OPENPI_ROOT=../openpi pytest tests/test_progress_head.py -q
OPENPI_ROOT=../openpi python scripts/estimate_memory.py
```

Patches were cut against `0cc8e355f7bac0976db1cc3139b1ff0379feea60`. The apply
script checks that, checks the tree is clean, and runs `git apply --check`
before touching anything. It detects an already-patched checkout and does
nothing, so re-running is safe.

---

## What the fork actually looks like

Quoted from the checkout, not from memory.

### `TrainConfig` — `src/openpi/training/config.py:567`

A frozen dataclass. The fields that decide memory:

```python
model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
weight_loader: weight_loaders.WeightLoader = ...
ema_decay: float | None = 0.99
freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
batch_size: int = 32
fsdp_devices: int = 1
```

and the derived filter every training script uses:

```python
@property
def trainable_filter(self) -> nnx.filterlib.Filter:
    return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
```

Configs are a module-level list `_CONFIGS`, looked up by name with
`get_config(name)`. `pi05_b1k` is at `config.py:757`:

```python
TrainConfig(
    name="pi05_b1k",
    model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
    data=LeRobotB1KDataConfig(
        repo_id="turning_on_radio",
        base_config=DataConfig(
            data_cls=_lerobot_compat.LeRobotDataset,
            dataset_root="/viscam/u/shiyuc/openpi/2026-challenge-demos/b1k/turning_on_radio",
            prompt_from_task=True,
            dataset_kwargs={"tolerance_s": 5e-4},
        ),
        robot_config_name="b1k/R1Pro",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    save_interval=10_000,
    num_train_steps=50_000,
    assets_base_dir="./outputs/assets",
    checkpoint_base_dir="./outputs/checkpoints",
)
```

Three things to notice:

- It is a **full fine-tune** — no `freeze_filter`, so `nnx.Nothing`, so
  everything trains.
- `ema_decay` is left at its `0.99` default, which keeps a second full copy of
  every parameter.
- `dataset_root` is a **Stanford cluster path** (`/viscam/u/shiyuc/...`). It will
  not exist on a rented box. Override it with
  `--data.base_config.dataset_root=<LOCAL_PATH>`; `scripts/train_cloud.sh` always
  passes one explicitly.
- `Pi0Config.action_dim` is left at its default of **32**, not 23. The model
  emits 32 and `B1KOutputs` slices to `robot_config.action_dim` (23) on the way
  out.

### `Pi0.__init__` — `src/openpi/models/pi0.py:67`

```python
def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
    super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
    self.pi05 = config.pi05
    paligemma_config = _gemma.get_config(config.paligemma_variant)
    action_expert_config = _gemma.get_config(config.action_expert_variant)
    llm = nnx_bridge.ToNNX(_gemma.Module(
        configs=[paligemma_config, action_expert_config],   # BOTH experts, one module
        embed_dtype=config.dtype, adarms=config.pi05,
    ))
    llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
    img = nnx_bridge.ToNNX(_siglip.Module(
        num_classes=paligemma_config.width, variant="So400m/14",   # hardcoded upstream
        pool_type="none", scan=True, dtype_mm=config.dtype,
    ))
    img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
    self.PaliGemma = nnx.Dict(llm=llm, img=img)
    self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
    if config.pi05:
        self.time_mlp_in  = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
    else:
        ...
    self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
```

The single most consequential line is `configs=[paligemma_config,
action_expert_config]`: **both experts live inside one `PaliGemma/llm` module.**
See "The freeze filter trap" below.

`variant="So400m/14"` was hardcoded. Our patch makes it `config.siglip_variant`
(default unchanged) so a test can build a model small enough to differentiate on
a CPU — otherwise even a "tiny" Pi0 carries a 415M-parameter vision tower.

### `Pi0.compute_loss` — `src/openpi/models/pi0.py:189` (upstream)

```python
def compute_loss(self, rng, observation, actions, *, train=False) -> at.Float[at.Array, "*b ah"]:
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    ...
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

Standard flow matching. Returns **per-(batch, horizon)** loss; both training
entry points reduce it with `jnp.mean`. Our patch preserves that return type
exactly and adds `compute_losses`, which returns the terms separately.

Confirming the warning about `prefix_out` — `sample_actions` (`pi0.py:261`):

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], ...)
assert prefix_out is None
```

A head on the prefix could not be evaluated at inference at all.

### `b1k_policy.py` — observations and actions

`B1KInputs.__call__` (`src/openpi/policies/b1k_policy.py:61`):

- `observation/state` is raw proprioception; `extract_state_from_proprio` slices
  it by the robot config's `proprio` groups and **sums the two gripper fingers
  into one width value** per gripper.
- Up to three cameras are read in registry order and named
  `("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`; missing views are
  zero-filled with `image_mask=False`.
- `_parse_image` converts float→uint8 and CHW→HWC, because LeRobot stores
  float32 CHW.

`B1KOutputs.__call__` is two lines:

```python
# Only return the first 23 dims.
return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
```

The R1Pro action layout (`src/openpi/configs/robots/b1k.py`, `action_dim=23`):
base 0–2, torso 3–6, left arm 7–13, left gripper 14, right arm 15–21, right
gripper 22. The `needs_delta_comp` groups (torso, both arms) get delta-action
transforms; base and grippers do not.

### `freeze_filter` and the nnx filters it accepts

Applied in exactly two places, in both `scripts/train.py` and
`scripts/b1k/train_b1k.py`:

```python
params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
...
opt_state=tx.init(params.filter(config.trainable_filter)),
...
diff_state = nnx.DiffState(0, config.trainable_filter)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(...)
```

So freezing does three things at once: casts to bf16, drops the parameter from
optimizer state, and drops it from the gradient. That is where the saving comes
from.

Accepted filters are ordinary `nnx.filterlib` filters — `nnx.Param`, `nnx.All`,
`nnx.Not`, `nnx.Nothing`, `nnx.Everything`, any callable `(path, value) -> bool`
— plus openpi's own `nnx_utils.PathRegex`, which joins the parameter path with
`/` and `fullmatch`es a regex against it.

### LoRA support for pi05

**There is none.** `_CONFIGS` contains exactly two LoRA configs:

| config | model |
|---|---|
| `pi0_libero_low_mem_finetune` | `Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")` — `pi05` not set |
| `pi0_fast_libero_low_mem_finetune` | `Pi0FASTConfig(..., paligemma_variant="gemma_2b_lora")` |

Nothing in the fork combines `pi05=True` with a `*_lora` variant. The machinery
is not pi0-specific — `gemma.get_config` returns `lora_configs` for
`gemma_2b_lora` (rank 16 on attn and ffn) and `gemma_300m_lora` (rank 32), and
`Block` threads them through `lora.Einsum`/`lora.FeedForward` regardless of
`adarms` — so it is expected to work. It is simply untested here, which matches
the state of upstream issue #672. Budget debugging time.

---

## Measured memory

`scripts/estimate_memory.py`, sizing the shipped configs from the real model
graph. STATE is resident training state — parameters, AdamW moments, EMA,
gradients — before a single activation.

| config | total | trainable | STATE | fits |
|---|---|---|---|---|
| `pi05_b1k` (stock, full FT) | 3.353B | 3.353B | **75.0 G** | 80GB card only, and barely |
| `pi05_b1k_frozen_vlm` | 3.353B | 0.430B | 13.5 G | RTX 4090 up |
| `pi05_b1k_lora` | 3.403B | 0.052B | 7.2 G | RTX 4090 up |
| `pi05_b1k_frozen_vlm_progress` | 3.353B | 0.430B | 13.5 G | RTX 4090 up |

75.0 G for the stock config independently reproduces openpi's own ">70 GB"
figure, which is a good sign the model of `init_train_state` is right.

**Expect Option A (`pi05_b1k_frozen_vlm`) to be the one that runs.** LoRA is
cheaper on paper but is the untested path; A is a filter change to code that
already works, and 13.5 G leaves ~27 G of activation headroom even on a 40 GB
card. Use LoRA if activations turn out to blow the budget at the batch size you
want, since `fsdp_devices` is the only other lever and it costs throughput.

Note that openpi has **no gradient accumulation** (grep: no `accum` anywhere in
`src/` or `scripts/`), so a smaller batch is a genuinely smaller effective
batch.

## The LoRA filter has the same hole

`Pi0Config.get_freeze_filter` only ever builds regexes over `.*llm.*`, and
nothing in the model puts LoRA in the vision tower. So a LoRA config as openpi
ships it leaves **all ~414M SigLIP parameters fully trainable**:

| filter | trainable | STATE |
|---|---|---|
| `cfg.get_freeze_filter()` (stock) | 0.467B | 14.2 G |
| `lora_freeze_filter(cfg)` | 0.052B | 7.2 G |

0.467B is *more* trainable weight than full-expert fine-tuning, which is not
what "LoRA" implies and not what the ">22.5 GB" row in openpi's README suggests
you are buying. `pi05_b1k_lora` uses `lora_freeze_filter`; pinned by
`test_lora_freeze_filter_covers_the_vision_tower`.

## The freeze filter trap

`nnx_utils.PathRegex(".*llm.*")` does **not** mean "the VLM". Measured on
`gemma_2b` + `gemma_300m` (`scripts/estimate_memory.py`):

| filter | frozen | trainable | what is actually trainable |
|---|---|---|---|
| `PathRegex(".*llm.*")` | 2.936B | 0.417B | **the SigLIP vision tower**, plus the projections |
| `PathRegex(".*llm.*_1.*")` | 0.428B | 2.925B | everything except the action expert |
| `freeze_vlm_filter()` | 2.923B | 0.430B | the action expert, plus the projections and any new head |

Both experts are inside `PaliGemma/llm`; only the action expert's leaves carry a
`_1` suffix (`gemma._name`, which exists so expert 0 loads straight from a
PaliGemma checkpoint). And `.*llm.*` never matches `PaliGemma/img`, so the
vision tower stays trainable under it.

The two wrong filters cost about the same memory as the right one, so a run
would look healthy and train the wrong 0.42B parameters. Our patch adds:

```python
VLM_PATHS = nnx_utils.PathRegex(".*(llm|img).*")
ACTION_EXPERT_PATHS = nnx_utils.PathRegex(".*llm.*_1.*")

def freeze_vlm_filter(*, freeze_image_tower: bool = True) -> nnx.filterlib.Filter:
    frozen = VLM_PATHS if freeze_image_tower else nnx_utils.PathRegex(".*llm.*")
    return nnx.All(frozen, nnx.Not(ACTION_EXPERT_PATHS))
```

pinned by `test_freeze_filter_regression_llm_regex_freezes_the_wrong_half`.

---

## The progress head

`Pi0.__init__` gains, alongside `action_out_proj` and reading the same tensor:

```python
self.progress_out_proj = nnx.Linear(hidden or action_expert_config.width, config.progress_dim, rngs=rngs)
```

`_progress_logits` mean-pools `suffix_out[:, -action_horizon:]` and projects.
Suffix, not prefix, for the reason quoted above. Mean-pooled off the final
output rather than probed per layer because `gemma.Module` builds its blocks
with `nn.scan(block_cls, variable_axes={"params": 0}, length=depth)`, which
stacks every layer's parameters on a leading axis and leaves no per-layer output
to attach to.

Loss is binary cross-entropy, so scalar progress (a soft target in [0,1]) and
the per-predicate vector (hard 0/1 per BDDL goal predicate) use the same head and
the same loss, and `progress_dim` can grow from 1 to P without a rewrite.

Config knobs on `Pi0Config`: `progress_head`, `progress_dim`,
`progress_hidden_dim`, `progress_loss_weight`.

`compute_loss` keeps its exact upstream signature and `*b ah` return type, so
both training entry points work untouched; `compute_losses` returns
`{action_loss, progress_loss, loss}` and `train_b1k.py` now logs the terms
separately. A fused number cannot tell you whether the head is learning or the
action term is just carrying the average down.

### Getting the label to the model

Three hops, each of which silently drops an unknown key:

1. `LeRobotB1KDataConfig.progress_key` → adds `"progress"` to the repack mapping.
2. `B1KInputs.__call__` → forwards `data["progress"]` when present.
3. `Observation.progress` → a new optional field, read by `Observation.from_dict`.

Hop 3 has a trap worth naming: `preprocess_observation` does **not** use
`dataclasses.replace`. It rebuilds `Observation` field by field, so any new field
not listed there is dropped between the data loader and the model, and the head
trains on nothing while the config says it is enabled. Our patch adds `progress`
to that constructor and a comment saying why.

#### Where the label comes from: a real dataset column

Traced, not assumed. `RepackTransform` maps flat SOURCE keys to model input
names, raises `KeyError` on a missing source key, and **discards everything not
named in the mapping**:

```python
def __call__(self, data: DataDict) -> DataDict:
    flat_item = flatten_dict(data)
    return jax.tree.map(lambda k: flat_item[k], self.structure)
```

and its input is literally the HF dataset row —
`dataset_reader.get_item` starts with `item = self.hf_dataset[idx]`. So
`progress_key` needs `progress` to exist as a dataset column. That settles
(a)-vs-(b): the merge-into-the-dataset route is what the mechanism was built for.

**The trap that makes it dangerous.** LeRobot loads parquet with an explicit
schema taken from `meta/info.json`:

```python
features = get_hf_features_from_features(self._meta.features)
hf_dataset = load_nested_dataset(self.root / "data", features=features, ...)
# -> datasets.Dataset.from_parquet(paths, features=features)
```

An extra parquet column that is **not** registered in `info.json` is not
silently dropped — it raises `CastError` → `DatasetGenerationError`, and the
dataset stops loading for *every* config, the baseline included. Verified
experimentally, and pinned by
`test_unregistered_column_would_break_the_whole_dataset`.

`scripts/merge_progress_labels.py` therefore writes the parquet column and the
`info.json` feature entry together, then reloads the result to prove it. It
writes a **new root** by default and symlinks `videos/` back to the original, so
the 330 GB copy is never mutated and the baseline arm provably reads the original
bytes.

The feature entry must be `{"dtype": "float32", "shape": [1]}`: LeRobot coerces
the shape list to a tuple on load, and `get_hf_features_from_features` maps
`(1,)` to `datasets.Value` — a scalar. Any other shape yields a length-1
sequence instead.

```bash
# on the cloud box, where the data lives
python scripts/merge_progress_labels.py \
    --dataset-root ~/data/b1k/turning_on_radio \
    --labels       ~/labels/turning_on_radio.parquet \
    --out-root     ~/data/b1k/turning_on_radio+progress

PROGRESS_KEY=progress CONFIG=pi05_b1k_frozen_vlm_progress \
DATASET_ROOT=~/data/b1k/turning_on_radio+progress bash scripts/train_cloud.sh
```

`progress_key` stays `None` in the shipped config until that column exists. In
that state the head is built and its parameters allocated, but the progress term
is skipped and training is identical to `pi05_b1k_frozen_vlm` — pinned by
`test_no_label_means_no_progress_term`, and warned about by `train_cloud.sh`.

### Loading a checkpoint with a new head

`state.replace_by_pure_dict` tolerates a partial dict, but the gate *before* it
does not. `CheckpointWeightLoader.load` calls
`_merge_params(loaded, params, missing_regex=".*lora.*")`, which returns the
intersection plus regex matches; `_load_weights_and_validate` then runs
`at.check_pytree_equality(expected=params_shape, got=loaded_params)`, which
raises on any key the merge dropped. A new head is not in `pi05_base` and does
not match `.*lora.*`, so **training aborts at startup** — after the ~7 GB base
checkpoint has downloaded.

Our patch makes `missing_regex` a field, and
`pi05_b1k_frozen_vlm_progress` sets it to `".*(lora|progress).*"`.

---

## What runs at step 0

Measured, not assumed (`test_pi05_starts_with_its_adarms_gates_closed`).

pi05 injects the flow timestep through adaptive RMSNorm, whose modulation is
`nn.Dense(kernel_init=nn.initializers.zeros)` (`gemma.RMSNorm`). So at
initialization every gate is zero, every residual branch in the action expert is
closed, and `suffix_out` **does not depend on the images or the prompt at all**.

Exactly six parameter groups receive gradient on the first step:

```
PaliGemma/llm/final_norm_1/Dense_0            (adaRMS modulation)
PaliGemma/llm/layers/pre_attention_norm_1/Dense_0
PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0
action_in_proj
action_out_proj
progress_out_proj
```

The vision tower, the whole PaliGemma LLM, the action expert's own attention and
MLP weights, and the time MLP all receive **exactly zero**. The gates open within
the first few steps and the rest starts training then.

This matters when reading early logs: a step-0 check on the expert's MLP
`grad_norm` reads as "broken model" and is not.
