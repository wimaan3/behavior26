#!/usr/bin/env bash
#
# Serve the released turning_on_radio baseline for EVALUATION.
#
# JAX MEMORY -- evaluation wants the OPPOSITE of training.
#
# HERE, SERVING: this process shares one GPU with Isaac Sim, which needs ~14 GB
# of a 24 GB 4090. JAX preallocates 75% of the card at init by default, which
# starves the simulator -- and the failure surfaces as a renderer error that
# never mentions JAX, so it is expensive to diagnose. Hence PREALLOCATE=false and
# a small fraction. 0.35 measured working: 19.4 GB total with both resident.
#
# IN scripts/train_cloud.sh, TRAINING: JAX owns the card. Preallocation stays ON
# and takes 90% -- faster, and it avoids arena fragmentation over a long run.
#
# Do not copy either setting to the other side. See AB_PROTOCOL revision
# 2026-09-14.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

OPENPI_ROOT="${OPENPI_ROOT:-/opt/openpi}"
CKPT="${CKPT:-/opt/baseline/extracted/pi05_turn_on_the_radio}"
TASK="${TASK:-turning_on_radio}"
PORT="${PORT:-8000}"

cd "${OPENPI_ROOT}"
# NO `policy:checkpoint` subcommand. The fork's docs/b1k.md shows one, but at
# commit 0cc8e355 Args.policy is a plain dataclass rather than a Union, so tyro
# rejects it. Flags are --policy.config / --policy.dir, and --repo-id is
# hyphenated. --repo-id must be the asset id the checkpoint ships its norm stats
# under (assets/turning_on_radio/), NOT the b1k/-prefixed task name that
# serve_b1k defaults to.
exec uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro \
  --task "b1k/${TASK}" \
  --repo-id "${TASK}" \
  --port "${PORT}" \
  --policy.config pi05_b1k \
  --policy.dir "${CKPT}"
