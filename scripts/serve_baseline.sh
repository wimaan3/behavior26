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

# uv installs to ~/.local/bin and install_openpi.sh only exported that inside its
# own shell, so a fresh shell gets "exec: uv: not found".
export PATH="${HOME}/.local/bin:${PATH}"

OPENPI_ROOT="${OPENPI_ROOT:-/opt/openpi}"
TASK="${TASK:-turning_on_radio}"
PORT="${PORT:-8000}"
# The RELEASED baseline trained under pi05_b1k. Our own arms train under
# pi05_b1k_frozen_vlm, and arm B's checkpoint carries a progress head that only the
# patched config declares -- so serving an arm needs the config it trained under:
#   CONFIG=pi05_b1k_frozen_vlm_progress CKPT=/workspace/shot1/checkpoints/pi05_b1k_frozen_vlm_progress/armB/9999 \
#   TASK=putting_shoes_on_rack ASSET_ID=set_up_a_coffee_station_in_your_kitchen bash serve_baseline.sh
# Hardcoding pi05_b1k here would fail only AFTER Isaac Sim had paid its scene load.
CONFIG="${CONFIG:-pi05_b1k}"
# Where the checkpoint's norm stats live: <checkpoint>/assets/<ASSET_ID>/. openpi takes
# this from --repo-id, and defaults it to the task. shot one trained coffee + shoes as ONE
# MultiLeRobotDataset, so one stats file covers both -- filed under the FIRST task's id.
# Evaluating shoes with ASSET_ID=putting_shoes_on_rack would look in a folder that does
# not exist. Set ASSET_ID to where the stats actually are; the task stays the task.
ASSET_ID="${ASSET_ID:-${TASK}}"
CKPT_ROOT="${CKPT_ROOT:-/opt/baseline}"
# DISCOVER the checkpoint rather than hardcode it: the directory name comes from
# inside the challenge's zip (pi05_turn_on_the_radio today) and is not ours to
# rely on. An orbax checkpoint is the directory CONTAINING params/, so find that.
CKPT="${CKPT:-$(find "${CKPT_ROOT}" -maxdepth 3 -type d -name params -printf '%h\n' 2>/dev/null | head -1)}"
[ -n "${CKPT}" ] && [ -d "${CKPT}/params" ] \
  || { echo "no checkpoint under ${CKPT_ROOT} (looked for a dir containing params/)" >&2; exit 1; }
echo "checkpoint: ${CKPT}"
echo "config: ${CONFIG}  task: ${TASK}  norm stats: assets/${ASSET_ID}"

# NORM STATS RIDE INSIDE THE CHECKPOINT. openpi reads them from
# <checkpoint>/assets/<asset id> and there is no flag to point elsewhere: Checkpoint
# carries only config and dir. When they are missing it logs "not found ... skipping"
# and serves an UNNORMALISED policy that scores near zero -- which, on an arm, reads
# as the treatment failing rather than as a missing file, after Isaac Sim has already
# paid a 12-minute scene load. So check now, when it costs nothing.
# The EXACT file openpi will load -- not "any stats file": that check would have passed a
# checkpoint whose stats sit under a different asset id, and served it unnormalised.
if [ ! -f "${CKPT}/assets/${ASSET_ID}/norm_stats.json" ]; then
  echo "no ${CKPT}/assets/${ASSET_ID}/norm_stats.json -- openpi would serve this policy \
UNNORMALISED and it would score ~0. Present: $(ls "${CKPT}/assets" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

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
  --repo-id "${ASSET_ID}" \
  --port "${PORT}" \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT}"
