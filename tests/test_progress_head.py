"""Prove the progress head is wired correctly, without a GPU.

This trains nothing. It answers the one question a rented 8xH100 should never be
the first thing to ask: does the gradient of the loss actually reach the new
parameters? A silently disconnected head -- reading a detached tensor, excluded
by a freeze filter, or dropped by a transform before the model sees its label --
costs a full run (~$185, ~8 hours) to discover, and looks like a healthy loss
curve while it does, because the action term dominates the average.

Models are built once per module: a tiny Pi0 still takes ~7s to construct and
~10s to differentiate on CPU.

    OPENPI_ROOT=../openpi pytest tests/test_progress_head.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.openpi_fixture import require_openpi, tiny_config

pytestmark = pytest.mark.openpi

BATCH = 2
PROGRESS_DIM = 3
LOSS_WEIGHT = 0.5


@pytest.fixture(scope="module")
def openpi():
    return require_openpi()


@pytest.fixture(scope="module")
def head_model(openpi):
    """Tiny pi05 with the progress head enabled."""
    import jax

    _, pi0_config, _, _ = openpi
    cfg = tiny_config(
        pi0_config, progress_head=True, progress_dim=PROGRESS_DIM, progress_loss_weight=LOSS_WEIGHT
    )
    return cfg, cfg.create(jax.random.key(0))


@pytest.fixture(scope="module")
def plain_model(openpi):
    """The same model with the head off -- the control."""
    import jax

    _, pi0_config, _, _ = openpi
    cfg = tiny_config(pi0_config)
    return cfg, cfg.create(jax.random.key(0))


@pytest.fixture(scope="module")
def abstract_head_model(openpi):
    """Shape-only model. Free to build, enough for filter questions."""
    import flax.nnx as nnx
    import jax

    _, pi0_config, _, _ = openpi
    cfg = tiny_config(
        pi0_config, progress_head=True, progress_dim=PROGRESS_DIM, progress_loss_weight=LOSS_WEIGHT
    )
    return cfg, nnx.eval_shape(lambda: cfg.create(jax.random.key(0)))


def _batch(openpi, cfg, *, progress=None):
    """A fake batch shaped the way the data loader delivers one."""
    import jax.numpy as jnp

    _, _, _model, _ = openpi
    obs = cfg.fake_obs(batch_size=BATCH)
    if progress is None:
        return obs, cfg.fake_act(batch_size=BATCH)
    # Observation is frozen and rebuilt field by field in several places, so
    # construct it explicitly rather than relying on dataclasses.replace.
    return (
        _model.Observation(
            images=obs.images,
            image_masks=obs.image_masks,
            state=obs.state,
            tokenized_prompt=obs.tokenized_prompt,
            tokenized_prompt_mask=obs.tokenized_prompt_mask,
            progress=jnp.asarray(progress, dtype=jnp.float32),
        ),
        cfg.fake_act(batch_size=BATCH),
    )


def _grads_by_path(model, obs, actions):
    """Gradient of the mean loss w.r.t. every parameter, keyed by joined path."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    def loss_fn(m):
        return jnp.mean(m.compute_losses(jax.random.key(2), obs, actions, train=False)["loss"])

    flat = tu.flatten_dict(nnx.grad(loss_fn)(model).to_pure_dict())
    return {"/".join(map(str, k)): np.asarray(v) for k, v in flat.items()}


@pytest.fixture(scope="module")
def grads_low(openpi, head_model):
    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=np.zeros((BATCH, PROGRESS_DIM)))
    return _grads_by_path(model, obs, actions)


@pytest.fixture(scope="module")
def grads_high(openpi, head_model):
    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=np.ones((BATCH, PROGRESS_DIM)))
    return _grads_by_path(model, obs, actions)


# --------------------------------------------------------------- loss shapes


def test_forward_and_loss_are_finite_with_the_head_on(openpi, head_model):
    import jax

    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=np.full((BATCH, PROGRESS_DIM), 0.25))
    losses = model.compute_losses(jax.random.key(1), obs, actions, train=False)

    assert set(losses) == {"action_loss", "progress_loss", "loss"}
    for name, value in losses.items():
        assert value.shape == (BATCH, cfg.action_horizon), f"{name} has shape {value.shape}"
        assert np.all(np.isfinite(np.asarray(value))), f"{name} is not finite"

    # The combined loss must be the weighted sum, not one of the parts.
    expected = np.asarray(losses["action_loss"]) + LOSS_WEIGHT * np.asarray(losses["progress_loss"])
    np.testing.assert_allclose(np.asarray(losses["loss"]), expected, rtol=1e-5)


def test_compute_loss_keeps_the_upstream_contract(openpi, head_model):
    """`compute_loss` must still return a bare `*b ah` array.

    scripts/train.py and scripts/b1k/train_b1k.py both call `jnp.mean` on its
    result. Returning a dict here would break every training entry point.
    """
    import jax

    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=np.full((BATCH, PROGRESS_DIM), 0.5))
    loss = model.compute_loss(jax.random.key(1), obs, actions, train=False)
    assert loss.shape == (BATCH, cfg.action_horizon)


def test_mismatched_target_width_fails_loudly(openpi, head_model):
    import jax

    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=np.full((BATCH, PROGRESS_DIM + 2), 0.5))
    with pytest.raises(ValueError, match="width 5"):
        model.compute_losses(jax.random.key(1), obs, actions, train=False)


# ----------------------------------------------------------------- gradients


def test_gradients_reach_the_progress_head(grads_low):
    """The assertion this file exists for."""
    head = {p: g for p, g in grads_low.items() if "progress" in p}
    assert head, f"no progress_head parameters found; got {sorted(grads_low)[:10]}"
    assert set(head) == {"progress_out_proj/kernel", "progress_out_proj/bias"}, sorted(head)
    for path, g in head.items():
        assert np.any(g != 0.0), f"gradient w.r.t. {path} is entirely zero -- the head is disconnected"
        assert np.all(np.isfinite(g)), f"gradient w.r.t. {path} is not finite"


def test_gradients_still_reach_the_action_decoder(grads_low):
    """Guards the test above: a harness that zeroed every gradient would pass it."""
    assert np.any(grads_low["action_out_proj/kernel"] != 0.0)


def test_the_progress_term_moves_the_shared_trunk(grads_low, grads_high):
    """The head must condition the trunk, not sit in a side channel.

    If the progress gradient stopped at progress_out_proj, the head would be a
    readout of features it cannot influence, and could never improve the
    representation the actions are decoded from -- which is the entire thesis.

    The witness is the action expert's adaRMS modulation Dense rather than its
    MLP, because of the initialization documented in
    test_pi05_starts_with_its_adarms_gates_closed: at step 0 the expert's own
    attention and MLP weights get exactly zero gradient, so they cannot
    distinguish anything.
    """
    trunk = "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel"
    assert trunk in grads_low, sorted(k for k in grads_low if "llm" in k)
    assert np.any(grads_low[trunk] != 0.0), "shared trunk received no gradient at all"
    assert not np.allclose(grads_low[trunk], grads_high[trunk]), (
        "changing only the progress label left the shared trunk gradient unchanged"
    )


def test_progress_label_does_not_leak_into_the_action_decoder(grads_low, grads_high):
    """The converse: action_out_proj must be untouched by the progress label.

    The two heads read the same tensor but must stay separate paths. If this
    fails, the terms are entangled and the A/B against pi05_b1k_frozen_vlm
    measures nothing.
    """
    np.testing.assert_allclose(
        grads_low["action_out_proj/kernel"], grads_high["action_out_proj/kernel"], rtol=0, atol=0
    )


def test_pi05_starts_with_its_adarms_gates_closed(grads_low):
    """Documents a property that otherwise reads as a broken model.

    pi05 injects the flow timestep through adaptive RMSNorm, whose modulation is
    `nn.Dense(kernel_init=zeros)` (gemma.RMSNorm). So at initialization every
    gate is 0, every residual branch in the action expert is closed, and
    `suffix_out` does not depend on the images or the prompt at all. Measured
    here, exactly six parameter groups receive gradient on step 0; the vision
    tower, the whole PaliGemma LLM, the expert's own attention/MLP weights and
    the time MLP all receive exactly zero.

    Consequences worth knowing before reading a training curve:
      * a step-0 sanity check on the expert's MLP grad_norm reads as "broken";
      * the gates open within the first few steps, once the modulation Denses
        move, and the rest of the network starts training then.

    If this ever fails, the initialization changed and every early-training
    diagnostic built on it needs rechecking.
    """
    nonzero = {p for p, g in grads_low.items() if np.any(g != 0.0)}
    assert nonzero == {
        "PaliGemma/llm/final_norm_1/Dense_0/bias",
        "PaliGemma/llm/final_norm_1/Dense_0/kernel",
        "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias",
        "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel",
        "PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias",
        "PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel",
        "action_in_proj/bias",
        "action_in_proj/kernel",
        "action_out_proj/bias",
        "action_out_proj/kernel",
        "progress_out_proj/bias",
        "progress_out_proj/kernel",
    }, sorted(nonzero)


# --------------------------------------------------------------- absent label


def test_no_label_means_no_progress_term(openpi, head_model):
    """An enabled head with no label must behave exactly like the head-off config.

    `progress_key` is None in the shipped config until the dataset carries the
    column, so this is the state the first run will actually be in.
    """
    import jax

    cfg, model = head_model
    obs, actions = _batch(openpi, cfg, progress=None)
    assert obs.progress is None

    losses = model.compute_losses(jax.random.key(1), obs, actions, train=False)
    assert "progress_loss" not in losses
    np.testing.assert_array_equal(np.asarray(losses["loss"]), np.asarray(losses["action_loss"]))


def test_head_off_matches_upstream_shape(openpi, plain_model):
    import jax

    cfg, model = plain_model
    obs, actions = _batch(openpi, cfg)
    losses = model.compute_losses(jax.random.key(1), obs, actions, train=False)
    assert set(losses) == {"action_loss", "loss"}
    assert model.compute_loss(jax.random.key(1), obs, actions, train=False).shape == (BATCH, cfg.action_horizon)


# ----------------------------------------------------------------- inference


def test_predict_progress_returns_probabilities(openpi, head_model):
    """Readable at inference. sample_actions asserts `prefix_out is None`, so a
    head hanging off the prefix could not be evaluated here at all."""
    import jax

    cfg, model = head_model
    obs, _ = _batch(openpi, cfg)
    p = np.asarray(model.predict_progress(jax.random.key(3), obs))
    assert p.shape == (BATCH, PROGRESS_DIM)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_sample_actions_still_works_with_the_head_on(openpi, head_model):
    import jax

    cfg, model = head_model
    obs, _ = _batch(openpi, cfg)
    actions = np.asarray(model.sample_actions(jax.random.key(4), obs, num_steps=2))
    assert actions.shape == (BATCH, cfg.action_horizon, cfg.action_dim)
    assert np.all(np.isfinite(actions))


# ------------------------------------------------------------ freeze filters


def _param_paths(model, filt):
    import flax.nnx as nnx
    import flax.traverse_util as tu

    return {"/".join(map(str, p)) for p in tu.flatten_dict(nnx.state(model, filt).to_pure_dict())}


def test_freeze_vlm_filter_leaves_the_head_and_expert_trainable(openpi, abstract_head_model):
    import flax.nnx as nnx

    _, pi0_config, _, _ = openpi
    _, model = abstract_head_model
    filt = pi0_config.freeze_vlm_filter()

    frozen = _param_paths(model, nnx.All(nnx.Param, filt))
    trainable = _param_paths(model, nnx.All(nnx.Param, nnx.Not(filt)))

    assert not any("progress" in p for p in frozen), "the progress head was frozen"
    assert any("progress" in p for p in trainable)
    # The action expert (the `_1`-suffixed leaves) must be trainable...
    assert any("_1" in p for p in trainable if "llm" in p)
    # ...and the VLM proper, vision tower included, must not be.
    assert any("img" in p for p in frozen)
    assert not any("img" in p for p in trainable)


def test_freeze_filter_regression_llm_regex_freezes_the_wrong_half(openpi, abstract_head_model):
    """Pins the trap that `PathRegex(".*llm.*")` sets.

    Both experts live inside `PaliGemma/llm` (only the action expert's leaves
    carry a `_1` suffix, see gemma._name), so that regex freezes the action
    expert and leaves the SigLIP tower trainable -- the opposite of "freeze the
    VLM, train the action expert". If this test fails, the fork restructured the
    module tree and freeze_vlm_filter needs rechecking.
    """
    import flax.nnx as nnx

    _, _, _, nnx_utils = openpi
    _, model = abstract_head_model

    naive = nnx_utils.PathRegex(".*llm.*")
    trainable = _param_paths(model, nnx.All(nnx.Param, nnx.Not(naive)))

    assert any("img" in p for p in trainable), "expected the naive regex to leave the vision tower trainable"
    assert not any("llm" in p and "_1" in p for p in trainable), (
        "expected the naive regex to freeze the action expert too"
    )


def test_lora_freeze_filter_covers_the_vision_tower(openpi):
    """openpi's own LoRA freeze filter leaves all ~414M SigLIP params trainable.

    `get_freeze_filter` only ever builds regexes over `.*llm.*`, and nothing in
    the model puts LoRA in the vision tower. Measured on the real variants, the
    stock filter trains 0.467B parameters -- more than full-expert fine-tuning --
    of which 0.414B is the vision tower. `lora_freeze_filter` fixes that; this
    pins both halves so a regression in either is visible.
    """
    import flax.nnx as nnx
    import jax

    _, pi0_config, _, _ = openpi
    cfg = tiny_config(
        pi0_config, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    )
    model = nnx.eval_shape(lambda: cfg.create(jax.random.key(0)))

    def trainable(filt):
        return _param_paths(model, nnx.All(nnx.Param, nnx.Not(filt)))

    stock = trainable(cfg.get_freeze_filter())
    fixed = trainable(pi0_config.lora_freeze_filter(cfg))

    assert any("img" in p for p in stock), "expected the stock LoRA filter to leave the tower trainable"
    assert not any("img" in p for p in fixed), "lora_freeze_filter did not cover the vision tower"
    # LoRA adapters themselves must stay trainable -- otherwise nothing trains.
    assert any("lora" in p for p in fixed), sorted(fixed)[:10]
    # And the action decoder, or the model cannot adapt its output at all.
    assert "action_out_proj/kernel" in fixed
