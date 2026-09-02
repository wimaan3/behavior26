#!/usr/bin/env python3
"""Per-GPU memory for an openpi TrainConfig, derived from the real model graph.

Builds each config abstractly (`nnx.eval_shape`, so no arrays are allocated),
applies the config's own `freeze_filter`, and sizes the training state the way
`init_train_state` actually lays it out. Nothing here is hand-copied from a
config file, so it stays correct when a config changes.

    python scripts/estimate_memory.py
    python scripts/estimate_memory.py --config pi05_b1k_frozen_vlm --show-params

What it covers, and what it cannot:

* Covered -- parameters, AdamW moments, EMA, gradients. This is the resident
  state, the part that is there before a single activation is allocated, and the
  part that decides whether a config fits at all.
* NOT covered -- activations, which scale with batch size and sequence length.
  gemma.Module wraps every block in `nn.remat(policy=nothing_saveable)`, so
  activation memory is small relative to the state but not zero. Treat the
  headroom column as the budget for it.

openpi has NO gradient accumulation (grep: no `accum` anywhere in src/ or
scripts/). Halving the batch size halves the effective batch, it does not just
make the step slower.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys

GIB = 1024**3

# The three configs this repo actually intends to run, plus the stock full
# fine-tune as the baseline that does not fit.
DEFAULT_CONFIGS = (
    "pi05_b1k",
    "pi05_b1k_frozen_vlm",
    "pi05_b1k_lora",
    "pi05_b1k_frozen_vlm_progress",
)

# Cards we might rent, by usable VRAM per device.
CARDS = (("RTX 4090", 24), ("A100 40GB", 40), ("A100 80GB", 80), ("H100 80GB", 80))


def _import_openpi():
    root = pathlib.Path(os.environ.get("OPENPI_ROOT", pathlib.Path(__file__).resolve().parents[2] / "openpi"))
    src = root / "src"
    if not (src / "openpi" / "training" / "config.py").exists():
        sys.exit(
            f"openpi checkout not found at {root}.\n"
            "Clone it as a sibling directory, or set OPENPI_ROOT:\n"
            "  git clone -b behavior https://github.com/wensi-ai/openpi.git"
        )
    sys.path.insert(0, str(src))
    # openpi_client is a local workspace package, not installed by a plain
    # `pip install jax flax`.
    sys.path.insert(0, str(root / "packages" / "openpi-client" / "src"))
    _stub_lerobot_if_unavailable()
    return root


def _stub_lerobot_if_unavailable() -> None:
    """Let this tool run without the LeRobot data stack.

    `openpi.training.config` imports `openpi.training.lerobot_compat`, which
    imports LeRobot's dataset classes purely to re-export them as type
    references (`data_cls: Any = _lerobot_compat.LeRobotDataset`). Pulling in
    that stack means lerobot + datasets + av + torchvision, none of which this
    tool touches: it sizes the model graph, not the data pipeline.

    On a box that ran `uv sync` the real import succeeds and this does nothing.
    """
    import importlib

    try:
        importlib.import_module("openpi.training.lerobot_compat")
        return
    except Exception as exc:  # noqa: BLE001 - any failure in that chain is the same to us
        reason = f"{type(exc).__name__}: {exc}"

    import types

    stub = types.ModuleType("openpi.training.lerobot_compat")
    for name in ("LeRobotDataset", "LeRobotDatasetMetadata", "MultiLeRobotDataset"):
        setattr(stub, name, type(name, (), {}))
    stub.tasks_from_metadata = lambda metadata: {}  # type: ignore[attr-defined]
    stub.__doc__ = "behavior26 stub -- see scripts/estimate_memory.py"
    sys.modules["openpi.training.lerobot_compat"] = stub

    print(
        f"note: LeRobot is unavailable ({reason}).\n"
        "      Using placeholder dataset classes. Parameter counts and memory are\n"
        "      unaffected -- they come from the model graph -- but nothing here\n"
        "      validates the data pipeline.",
        file=sys.stderr,
    )


@dataclasses.dataclass(frozen=True)
class Sizing:
    name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    ema: bool

    @property
    def param_bytes(self) -> int:
        # init_train_state casts frozen params to bf16; trainable stay fp32.
        return self.trainable_params * 4 + self.frozen_params * 2

    @property
    def optimizer_bytes(self) -> int:
        # optax.adamw with the default mu_dtype keeps mu and nu at the param
        # dtype, and tx.init only ever sees the trainable subset.
        return self.trainable_params * 8

    @property
    def ema_bytes(self) -> int:
        # ema_params starts as a copy of the full params state, frozen half
        # included -- which is why every low-memory config turns EMA off.
        return self.param_bytes if self.ema else 0

    @property
    def grad_bytes(self) -> int:
        # nnx.DiffState(0, trainable_filter): gradients exist for trainables only.
        return self.trainable_params * 4

    @property
    def update_bytes(self) -> int:
        # optax.apply_updates allocates one more trainable-sized buffer.
        return self.trainable_params * 4

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.optimizer_bytes + self.ema_bytes + self.grad_bytes + self.update_bytes


def measure(name: str) -> Sizing:
    import flax.nnx as nnx
    import jax
    import numpy as np

    import openpi.training.config as _config

    cfg = _config.get_config(name)
    model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))

    def count(filt) -> int:
        state = nnx.state(model, filt)
        return sum(int(np.prod(v.shape)) for v in jax.tree.leaves(state.to_pure_dict()))

    total = count(nnx.Param)
    trainable = count(cfg.trainable_filter)
    return Sizing(
        name=name,
        total_params=total,
        trainable_params=trainable,
        frozen_params=total - trainable,
        ema=cfg.ema_decay is not None,
    )


def show_params(name: str) -> None:
    """Every trainable parameter group, largest first -- for auditing a filter."""
    import collections

    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import numpy as np

    import openpi.training.config as _config

    cfg = _config.get_config(name)
    model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    groups: dict[str, int] = collections.defaultdict(int)
    for path, leaf in tu.flatten_dict(nnx.state(model, cfg.trainable_filter).to_pure_dict()).items():
        joined = "/".join(map(str, path))
        groups["/".join(joined.split("/")[:3])] += int(np.prod(leaf.shape))

    print(f"\ntrainable parameter groups for {name}:")
    for group, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {n / 1e6:10.1f}M  {group}")
    if not groups:
        print("  (none -- every parameter is frozen)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", help="config name; repeatable. Defaults to the behavior26 set.")
    ap.add_argument("--show-params", action="store_true", help="also list trainable parameter groups")
    args = ap.parse_args()

    _import_openpi()
    names = args.config or list(DEFAULT_CONFIGS)

    sizings = []
    for name in names:
        try:
            sizings.append(measure(name))
        except Exception as exc:  # noqa: BLE001 - one bad config must not hide the rest
            print(f"!! {name}: could not size ({type(exc).__name__}: {exc})", file=sys.stderr)

    if not sizings:
        return 1

    header = f"{'config':32s} {'total':>8s} {'train':>8s} {'params':>8s} {'optim':>7s} {'ema':>6s} {'grad':>6s} {'STATE':>8s}"
    print(header)
    print("-" * len(header))
    for s in sizings:
        print(
            f"{s.name:32s} {s.total_params / 1e9:7.3f}B {s.trainable_params / 1e9:7.3f}B "
            f"{s.param_bytes / GIB:7.1f}G {s.optimizer_bytes / GIB:6.1f}G {s.ema_bytes / GIB:5.1f}G "
            f"{(s.grad_bytes + s.update_bytes) / GIB:5.1f}G {s.total_bytes / GIB:7.1f}G"
        )

    print("\nSTATE is resident training state only. Activations are extra and scale with batch size.")
    print("Fit assessment (headroom = card VRAM - STATE, i.e. the activation budget):\n")
    for s in sizings:
        gib = s.total_bytes / GIB
        verdicts = []
        for card, vram in CARDS:
            head = vram - gib
            mark = "no" if head <= 0 else ("tight" if head < 0.25 * vram else "yes")
            verdicts.append(f"{card}: {mark} ({head:+.0f}G)")
        print(f"  {s.name:32s} {gib:6.1f}G  " + "  ".join(verdicts))

    if args.show_params:
        for name in names:
            show_params(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
