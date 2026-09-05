"""Verify the SHIPPED training configs, not the helpers they are supposed to use.

The existing tests in test_progress_head.py check that `freeze_vlm_filter` and
`lora_freeze_filter` behave correctly. That is not the same as checking that the
configs actually call them -- a config could use the wrong filter and every one
of those tests would still pass.

Both failures here cost a paid run:

  * the wrong freeze filter trains the vision tower instead of the action expert,
    at the same memory cost, so nothing looks wrong until the numbers come back;
  * a `missing_regex` that does not cover the progress head aborts at startup,
    after the ~7 GB base checkpoint has downloaded.

    OPENPI_ROOT=../openpi pytest tests/test_shipped_configs.py -q
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.openpi_fixture import require_openpi

pytestmark = pytest.mark.openpi

FROZEN_VLM_CONFIGS = ["pi05_b1k_frozen_vlm", "pi05_b1k_frozen_vlm_progress"]
PROGRESS_CONFIGS = ["pi05_b1k_frozen_vlm_progress"]
ALL_B1K = [*FROZEN_VLM_CONFIGS, "pi05_b1k_lora"]


@pytest.fixture(scope="module")
def openpi():
    return require_openpi()


@pytest.fixture(scope="module")
def configs(openpi):
    """Import openpi.training.config, stubbing LeRobot if its video stack is broken."""
    import importlib
    import sys
    import types

    try:
        importlib.import_module("openpi.training.lerobot_compat")
    except Exception:  # noqa: BLE001 - lerobot pulls in av/torchvision; we need neither
        stub = types.ModuleType("openpi.training.lerobot_compat")
        for name in ("LeRobotDataset", "LeRobotDatasetMetadata", "MultiLeRobotDataset"):
            setattr(stub, name, type(name, (), {}))
        stub.tasks_from_metadata = lambda metadata: {}
        sys.modules["openpi.training.lerobot_compat"] = stub

    try:
        return importlib.import_module("openpi.training.config")
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"openpi.training.config not importable: {exc}")


def _param_split(cfg):
    """(frozen paths, trainable paths) for a config, built abstractly."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax

    model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))

    def paths(filt):
        flat = tu.flatten_dict(nnx.state(model, filt).to_pure_dict())
        return {"/".join(map(str, k)): v for k, v in flat.items()}

    return paths(nnx.All(nnx.Param, cfg.freeze_filter)), paths(cfg.trainable_filter)


def _count(params) -> int:
    return sum(int(np.prod(v.shape)) for v in params.values())


# ------------------------------------------------------------------- TASK 3


@pytest.mark.parametrize("name", FROZEN_VLM_CONFIGS)
def test_frozen_vlm_configs_train_the_action_expert_not_the_vision_tower(configs, name):
    """The freeze filter must do what its name says, in the shipped config.

    PathRegex(".*llm.*") would freeze the action expert and leave the 415M SigLIP
    tower trainable -- the opposite intent at the same memory cost.
    """
    cfg = configs.get_config(name)
    frozen, trainable = _param_split(cfg)

    # The action expert is the `_1`-suffixed half of PaliGemma/llm.
    expert = {p for p in trainable if "llm" in p and "_1" in p}
    assert expert, f"{name}: the action expert is not trainable -- wrong freeze filter"

    # The vision tower must be frozen, entirely.
    assert not any("img" in p for p in trainable), (
        f"{name}: SigLIP is trainable ({sorted(p for p in trainable if 'img' in p)[:3]}...). "
        'This is the PathRegex(".*llm.*") trap.'
    )
    assert any("img" in p for p in frozen), f"{name}: vision tower is not frozen"

    # And the non-expert half of the LLM must be frozen.
    assert not any("llm" in p and "_1" not in p for p in trainable), (
        f"{name}: part of the PaliGemma LLM proper is trainable"
    )


@pytest.mark.parametrize("name", FROZEN_VLM_CONFIGS)
def test_frozen_vlm_configs_match_the_documented_trainable_count(configs, name):
    """~0.430B, the figure README.md and training/README.md both quote.

    A config that quietly started training the vision tower would land near
    0.417B or 0.845B instead -- close enough to look plausible in a log line.
    """
    cfg = configs.get_config(name)
    _, trainable = _param_split(cfg)
    billions = _count(trainable) / 1e9
    assert 0.42 <= billions <= 0.44, f"{name}: {billions:.3f}B trainable, expected ~0.430B"


def test_lora_config_freezes_the_vision_tower_too(configs):
    """Stock get_freeze_filter only covers `.*llm.*`, so LoRA alone leaves SigLIP
    fully trainable -- 0.467B, more than full-expert fine-tuning."""
    cfg = configs.get_config("pi05_b1k_lora")
    frozen, trainable = _param_split(cfg)

    assert not any("img" in p for p in trainable), (
        "pi05_b1k_lora trains the vision tower; it should use pi0_config.lora_freeze_filter()"
    )
    assert any("lora" in p for p in trainable), "no LoRA adapters are trainable"
    billions = _count(trainable) / 1e9
    assert billions < 0.1, f"pi05_b1k_lora: {billions:.3f}B trainable, expected ~0.052B"


@pytest.mark.parametrize("name", ALL_B1K)
def test_partially_frozen_configs_disable_ema(configs, name):
    """EMA keeps an fp32 copy of ALL params and undoes most of the saving."""
    cfg = configs.get_config(name)
    assert cfg.ema_decay is None, f"{name}: ema_decay={cfg.ema_decay}, expected None"


# ------------------------------------------------------------------- TASK 2


@pytest.mark.parametrize("name", PROGRESS_CONFIGS)
def test_progress_head_params_are_matched_by_missing_regex(configs, name):
    """Every new parameter must match the loader's missing_regex.

    `_merge_params` returns the checkpoint/model intersection plus regex matches;
    `_load_weights_and_validate` then asserts structural equality against the
    full model params. Anything the merge dropped raises -- at startup, after the
    ~7 GB download.
    """
    cfg = configs.get_config(name)
    _, trainable = _param_split(cfg)

    new_params = sorted(p for p in trainable if "progress" in p)
    assert new_params, f"{name}: no progress head parameters found"

    regex = getattr(cfg.weight_loader, "missing_regex", None)
    assert regex is not None, f"{name}: weight loader has no missing_regex field"

    pattern = re.compile(regex)
    for path in new_params:
        assert pattern.fullmatch(path), (
            f"{name}: {path!r} is not matched by missing_regex {regex!r}. "
            "Training would abort in check_pytree_equality after the base checkpoint downloads."
        )


def test_missing_regex_gate_reproduced_end_to_end(configs):
    """Drive openpi's real merge + validate path with a checkpoint that lacks the head.

    This is the actual startup sequence, not a regex unit test: it proves the
    configured regex survives `_merge_params` followed by `check_pytree_equality`.
    """
    import flax.nnx as nnx
    import jax

    import openpi.shared.array_typing as at
    from openpi.training import weight_loaders

    cfg = configs.get_config("pi05_b1k_frozen_vlm_progress")
    model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    params = nnx.state(model, nnx.Param).to_pure_dict()

    # A "checkpoint" shaped like pi05_base: everything except the new head.
    import flax.traverse_util as tu

    flat = tu.flatten_dict(params, sep="/")
    checkpoint = tu.unflatten_dict({k: v for k, v in flat.items() if "progress" not in k}, sep="/")

    merged = weight_loaders._merge_params(checkpoint, params, missing_regex=cfg.weight_loader.missing_regex)
    at.check_pytree_equality(expected=params, got=merged, check_shapes=True)  # raises if a key was dropped

    # And confirm the guard is real: the stock regex must fail here.
    with pytest.raises(ValueError, match="different structure"):
        stock = weight_loaders._merge_params(checkpoint, params, missing_regex=".*lora.*")
        at.check_pytree_equality(expected=params, got=stock, check_shapes=True)
