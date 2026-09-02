#!/usr/bin/env bash
# Cold rented box -> checkpoint. Idempotent and resumable.
#
# One run is ~8 hours on 8xH100 (~$185 at the organizers' published figures), so
# every avoidable failure here is money. The design rules follow from that:
#
#   * Preflight everything cheap BEFORE anything expensive. Every check below
#     fails in seconds; the things they guard fail in minutes or hours.
#   * Never repeat the install on a retry. Each stage drops a stamp file in
#     $STATE_DIR; a stage with a stamp is skipped. Delete a stamp to force it.
#   * Never re-download the dataset. --dataset-root is always a LOCAL path.
#   * Push the checkpoint somewhere durable before the box is destroyed.
#
# Usage:
#   scripts/train_cloud.sh                          # defaults below
#   CONFIG=pi05_b1k_lora scripts/train_cloud.sh
#   scripts/train_cloud.sh --dry-run                # print the plan, run nothing
#   FORCE=norm_stats scripts/train_cloud.sh         # redo one stage
#
# Environment:
#   CONFIG          openpi train config name        (pi05_b1k_frozen_vlm)
#   EXP_NAME        experiment/checkpoint name      (<config>-<UTC date>)
#   OPENPI_ROOT     openpi checkout                 ($HOME/openpi)
#   DATASET_ROOT    LOCAL dataset dir, must exist   ($HOME/data/b1k/turning_on_radio)
#   NUM_GPUS        devices to use                  (all visible)
#   BATCH_SIZE      global batch                    (32)
#   NUM_TRAIN_STEPS                                 (30000)
#   CHECKPOINT_DEST rsync/rclone/gs destination for the finished checkpoint
#   WANDB_API_KEY   if unset, training runs with wandb disabled
set -euo pipefail

CONFIG="${CONFIG:-pi05_b1k_frozen_vlm}"
EXP_NAME="${EXP_NAME:-${CONFIG}-$(date -u +%Y%m%d)}"
OPENPI_ROOT="${OPENPI_ROOT:-$HOME/openpi}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/b1k/turning_on_radio}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
CHECKPOINT_DEST="${CHECKPOINT_DEST:-}"
FORCE="${FORCE:-}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

STATE_DIR="${OPENPI_ROOT}/.train_cloud_state/${CONFIG}"
LOG_DIR="${OPENPI_ROOT}/outputs/logs/${CONFIG}/${EXP_NAME}"
CKPT_DIR="${OPENPI_ROOT}/outputs/checkpoints/${CONFIG}/${EXP_NAME}"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" = 1 ]; then printf '    [dry-run] %s\n' "$*"; else "$@"; fi
}

# A stage runs once. Its stamp records the command line that produced it, so a
# changed config re-runs the stage instead of silently reusing a stale artifact.
stage() {
  local name="$1"; shift
  local stamp="${STATE_DIR}/${name}.done"
  local sig; sig="$(printf '%s ' "$@")"
  if [ "$FORCE" = "$name" ] || [ "$FORCE" = "all" ]; then
    rm -f "$stamp"
  elif [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sig" ]; then
    log "skip ${name} (already done)"; return 0
  elif [ -f "$stamp" ]; then
    warn "${name} was done with different arguments; re-running"
  fi
  log "${name}"
  run "$@"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$STATE_DIR"; printf '%s' "$sig" > "$stamp"; }
}

# ---------------------------------------------------------------- preflight
# Everything here is seconds. Everything it guards is minutes to hours.

log "preflight"

[ -d "$OPENPI_ROOT/src/openpi" ] || die "OPENPI_ROOT=$OPENPI_ROOT is not an openpi checkout"
cd "$OPENPI_ROOT"

command -v uv >/dev/null 2>&1 || die "uv not found -- openpi's scripts are invoked through it"
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi; wrong instance type?"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || die "nvidia-smi failed"

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPUS}"
[ "$NUM_GPUS" -ge 1 ] || die "NUM_GPUS=$NUM_GPUS"
# openpi asserts this itself, but it does so after building the model and
# downloading the base checkpoint. Fail in the first second instead.
[ $((BATCH_SIZE % NUM_GPUS)) -eq 0 ] \
  || die "BATCH_SIZE=$BATCH_SIZE must divide evenly across NUM_GPUS=$NUM_GPUS"

# The config must exist and must be one that fits. estimate_memory.py builds it
# abstractly, so this also catches a config that cannot even be constructed --
# a typo'd freeze filter, a bad variant name -- before any GPU work.
uv run python -c "
import openpi.training.config as c; cfg = c.get_config('${CONFIG}')
assert cfg.ema_decay is None or cfg.freeze_filter.__class__.__name__ == 'Nothing', (
    'ema_decay is on for a partially-frozen config: EMA keeps an fp32 copy of ALL '
    'params and undoes most of the memory saving. Set ema_decay=None.')
print(f'config {cfg.name}: batch={cfg.batch_size} steps={cfg.num_train_steps} ema={cfg.ema_decay}')
" || die "config '${CONFIG}' failed to load"

# msgpack-numpy: the policy server encodes ndarray observations with it. Absent,
# serving fails at the first observation -- after training has already finished.
uv run python -c "import msgpack_numpy" 2>/dev/null \
  || die "msgpack-numpy missing. It is needed to SERVE this checkpoint; install it now, not after the run: uv pip install msgpack-numpy"

# The pinned BEHAVIOR tag moves (v3.9.1 -> v3.9.2 already). Report the tag this
# box will evaluate against; a mismatch invalidates the numbers, not the run.
log "BEHAVIOR tag on this box: ${BEHAVIOR_TAG:-<unset>} -- confirm against
    https://behavior.stanford.edu/challenge/evaluation.html"

# Dataset must be LOCAL and non-empty. A network mount here turns an 8-hour run
# into a 30-hour one and the symptom is just "training is slow".
[ -d "$DATASET_ROOT" ] || die "DATASET_ROOT=$DATASET_ROOT does not exist. Fetch the data first (scripts/download_data.sh)."
[ -n "$(ls -A "$DATASET_ROOT" 2>/dev/null)" ] || die "DATASET_ROOT=$DATASET_ROOT is empty"
DATASET_FS="$(df -PT "$DATASET_ROOT" | awk 'NR==2 {print $2}')"
case "$DATASET_FS" in
  nfs*|cifs|fuse*|9p) warn "DATASET_ROOT is on a ${DATASET_FS} mount. Copy it to local disk -- data loading will dominate the run." ;;
esac
log "dataset: $DATASET_ROOT ($(du -sh "$DATASET_ROOT" 2>/dev/null | cut -f1), ${DATASET_FS})"

# Disk headroom for checkpoints. A 3.4B-param model is ~13GB per fp32 save.
AVAIL_GB="$(df -PBG "$OPENPI_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')"
[ "$AVAIL_GB" -ge 60 ] || warn "only ${AVAIL_GB}GB free under $OPENPI_ROOT; checkpoints are ~13GB each"

if [ -z "${WANDB_API_KEY:-}" ]; then
  warn "WANDB_API_KEY unset -- running with --nowandb_enabled. You will have no loss curve."
  WANDB_FLAG=(--nowandb_enabled)
else
  WANDB_FLAG=()
fi

if [ -z "$CHECKPOINT_DEST" ]; then
  warn "CHECKPOINT_DEST unset -- the checkpoint stays on this box and dies with it."
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
log "plan: config=${CONFIG} exp=${EXP_NAME} gpus=${NUM_GPUS} batch=${BATCH_SIZE} steps=${NUM_TRAIN_STEPS}"

# ------------------------------------------------------------------- install
# Guarded by a stamp: a retry after a training failure must not re-sync deps.

stage install uv sync

# --------------------------------------------------------------- norm stats
# Training aborts with a missing-norm-stats error if this is skipped, but only
# after building the model, so do it first.

stage norm_stats \
  uv run scripts/compute_norm_stats.py \
    --config-name "$CONFIG" \
    --data.base_config.dataset_root "$DATASET_ROOT"

# ------------------------------------------------------------------ training
# Not stamped: --resume makes re-running the correct recovery from a crash, and
# openpi refuses --resume together with --overwrite.

RESUME_FLAG=(--overwrite)
if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
  log "existing checkpoints in $CKPT_DIR -- resuming"
  RESUME_FLAG=(--resume)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

log "training -> $CKPT_DIR (log: $LOG_DIR/train.log)"
run uv run scripts/b1k/train_b1k.py "$CONFIG" \
    --exp_name="$EXP_NAME" \
    --batch_size="$BATCH_SIZE" \
    --num_train_steps="$NUM_TRAIN_STEPS" \
    --data.base_config.dataset_root "$DATASET_ROOT" \
    "${RESUME_FLAG[@]}" \
    "${WANDB_FLAG[@]}" \
    2>&1 | tee -a "$LOG_DIR/train.log"

[ "$DRY_RUN" = 1 ] || [ -d "$CKPT_DIR" ] || die "training finished but $CKPT_DIR does not exist"

# ------------------------------------------------------------------- publish

if [ -n "$CHECKPOINT_DEST" ]; then
  log "publishing $CKPT_DIR -> $CHECKPOINT_DEST"
  case "$CHECKPOINT_DEST" in
    gs://*) run gsutil -m rsync -r "$CKPT_DIR" "${CHECKPOINT_DEST%/}/${CONFIG}/${EXP_NAME}" ;;
    s3://*) run aws s3 sync "$CKPT_DIR" "${CHECKPOINT_DEST%/}/${CONFIG}/${EXP_NAME}" ;;
    *)      run rsync -av --partial "$CKPT_DIR/" "${CHECKPOINT_DEST%/}/${CONFIG}/${EXP_NAME}/" ;;
  esac
  log "published"
else
  warn "CHECKPOINT_DEST unset -- $CKPT_DIR is NOT backed up. Copy it off this box before destroying it."
fi

log "done: $CKPT_DIR"
