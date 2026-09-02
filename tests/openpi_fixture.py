"""Locate the openpi fork and build tiny models for CPU tests.

The fork is a sibling checkout, not a submodule and not on PyPI, so tests that
need it locate it at import time and skip cleanly when it is absent. A laptop
without jax must still be able to run the rest of the suite.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

# Sibling of the repo root by default; override for a checkout elsewhere.
DEFAULT_OPENPI = pathlib.Path(__file__).resolve().parents[2] / "openpi"


def openpi_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("OPENPI_ROOT", str(DEFAULT_OPENPI))).expanduser()


def require_openpi():
    """Import openpi from the sibling checkout, or skip the test with the reason.

    Returns the (pi0, pi0_config, model, nnx_utils) modules.
    """
    root = openpi_root()
    src = root / "src"
    if not (src / "openpi" / "models" / "pi0.py").exists():
        pytest.skip(
            f"openpi checkout not found at {root}. "
            "Clone it as a sibling directory or set OPENPI_ROOT:\n"
            "  git clone -b behavior https://github.com/wensi-ai/openpi.git"
        )
    for extra in (src, root / "packages" / "openpi-client" / "src"):
        # openpi_client is a local workspace package, not installed by a
        # plain `pip install jax flax`.
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        from openpi.models import model as _model  # noqa: PLC0415
        from openpi.models import pi0 as _pi0  # noqa: PLC0415
        from openpi.models import pi0_config as _pi0_config  # noqa: PLC0415
        from openpi.shared import nnx_utils as _nnx_utils  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"openpi found at {root} but not importable ({exc}). Its deps are not installed here.")
    return _pi0, _pi0_config, _model, _nnx_utils


def tiny_config(pi0_config, **overrides):
    """A Pi0Config small enough to differentiate on a CPU in a couple of seconds.

    `siglip_variant="mu/16"` is the point of the whole thing: the vision tower is
    otherwise hardcoded to So400m/14 (~415M params), so a "tiny" Pi0 built only
    from the dummy gemma variants would still be far too large to backprop here.

    Nothing this produces is trainable in any useful sense. It exists to check
    wiring: shapes line up, the loss is finite, and gradients reach the new head.
    """
    kwargs = {
        "pi05": True,
        "action_dim": 8,
        "action_horizon": 4,
        "max_token_len": 16,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "siglip_variant": "mu/16",
        "dtype": "float32",
    }
    kwargs.update(overrides)
    return pi0_config.Pi0Config(**kwargs)
