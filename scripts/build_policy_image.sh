#!/usr/bin/env bash
# Build the submission policy image (docker/policy/Dockerfile) from a local checkpoint.
#   CKPT=~/behavior26-checkpoints/shot2/armB/5999 TAG=behavior26-policy:armB-5999 bash scripts/build_policy_image.sh
# The checkpoint is staged (hard links where possible) into a context outside the repo, so
# the 6 GB of weights never touch git. Optimizer state (train_state/) is left out.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:?set CKPT to a checkpoint step directory (contains params/ and assets/)}"
TAG="${TAG:-behavior26-policy:armB-5999}"
CTX="${CTX:-$HOME/behavior26-docker-context}"
[ -d "$CKPT/params" ] && [ -f "$CKPT/assets/turning_on_radio/norm_stats.json" ] \
  || { echo "need $CKPT/params and $CKPT/assets/turning_on_radio/norm_stats.json"; exit 1; }
rm -rf "$CTX" && mkdir -p "$CTX/checkpoint" "$CTX/scripts" "$CTX/training"
cp -al "$CKPT/params" "$CKPT/assets" "$CTX/checkpoint/" 2>/dev/null || cp -a "$CKPT/params" "$CKPT/assets" "$CTX/checkpoint/"
[ -f "$CKPT/_CHECKPOINT_METADATA" ] && cp "$CKPT/_CHECKPOINT_METADATA" "$CTX/checkpoint/"
cp "$REPO/scripts/apply_openpi_patches.sh" "$CTX/scripts/"
cp -r "$REPO/training/patches" "$CTX/training/"
cp "$REPO/docker/policy/Dockerfile" "$CTX/Dockerfile"
echo "context: $CTX ($(du -sh "$CTX" | cut -f1))"
docker build -t "$TAG" "$CTX"
docker image inspect "$TAG" --format 'IMAGE_OK {{.Id}} {{.Size}}'
