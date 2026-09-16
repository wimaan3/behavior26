#!/usr/bin/env bash
#
# SHOT ONE -- the A/B the project exists to run.
#
#   arm A  pi05_b1k_frozen_vlm, no progress head        (control)
#   arm B  the same, plus the progress head at LAMBDA   (treatment)
#
# The arms must differ in the head and in NOTHING else. That is enforced here by
# construction rather than by care: one dataset root, one assets dir, one norm-stats
# computation shared by both, the same STEPS, the same seed, --protocol on both.
# A ΔQ from arms that differed in something else still looks like a result.
#
# COST. Rung 4 measured 3.90 s/step median and 8.11 s/step mean at 8 workers on an
# RTX PRO 4500, with 52.8% of the clock inside a prefetch sawtooth. So one arm is
# 22-68 h depending on STEPS and on whether WORKERS is set high enough to remove the
# stall. Read docs/sessionB-2026-09-16-rung2-5/README.md before choosing STEPS.
#
# SURVIVING THE POD. A run this long WILL be interrupted. Training checkpoints every
# SAVE_EVERY steps and re-invoking this script resumes: a finished arm is skipped,
# an unfinished one continues from its last checkpoint. Re-invocation costs the
# remaining work, not all of it.
#
# CREDENTIALS. The box is rented. Nothing here writes a token to disk or logs in.
# Pushing reads HF_TOKEN from the environment of whoever invokes the script, and the
# check for it happens BEFORE training rather than after 68 h of it.
#
set -uo pipefail

# --- what to run -------------------------------------------------------------
STEPS="${STEPS:-30000}"
# Rung 3 calibrated this: at lambda 0.1 the progress term is ~14-15% of the update,
# so ~0.14-0.15 buys 20%. It is a parameter, not a literal, so it cannot outlive the
# calibration it came from -- see docs/sessionB-2026-09-16-rung2-5/RUNG2-5.md.
LAMBDA="${LAMBDA:-0.15}"
SEED="${SEED:-42}"          # TrainConfig's own default; both arms share it
BATCH="${BATCH:-32}"
# Rung 4's 8 was TrainConfig's default, chosen to make throughput representative.
# Shot one wants throughput, not representativeness: the sawtooth is a worker-count
# problem and this is the lever. Check the box's vCPU count before raising it.
WORKERS="${WORKERS:-24}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
TASKS_CSV="${TASKS_CSV:-set_up_a_coffee_station_in_your_kitchen}"
STATS_FRAMES="${STATS_FRAMES:-128000}"
PUSH="${PUSH:-0}"                       # 1 = push checkpoints to HF_REPO
HF_REPO="${HF_REPO:-}"

# --- where things live -------------------------------------------------------
ROOT="${ROOT:-/opt/merged}"
ASSETS="${ASSETS:-/opt/assets_shot1}"
CKPT="${CKPT:-/opt/checkpoints}"
OUT="${OUT:-/opt/shot1}"
B26=/opt/behavior26
PY=/opt/openpi/.venv/bin/python
export OPENPI_ROOT=/opt/openpi
export PATH="${HOME}/.local/bin:${PATH}" HF_HUB_DISABLE_PROGRESS_BARS=1
# Training, not serving: preallocate. (Evaluation needs the opposite, because Isaac
# Sim wants the same GPU -- see docs/GOTCHAS.md.)
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

read -r -a TASKS <<< "${TASKS_CSV//,/ }"
mkdir -p "$OUT"
CUR=init
stage () { echo; echo "=== STAGE $1 $(date -u +%H:%M:%S)"; CUR="$1"; }
fail  () { echo "SHOT1_FAILED_AT=${CUR}: $*"; exit 1; }
TSTAMP='while IFS= read -r l; do printf "%s %s\n" "$(date +%s.%N)" "$l"; done'

# A single task means ROOT IS the dataset; several means ROOT is their parent.
if [ "${#TASKS[@]}" -eq 1 ]; then DATA_ROOT="$ROOT/${TASKS[0]}"; else DATA_ROOT="$ROOT"; fi

# --- 0 preflight: everything that can fail cheaply, before anything expensive --
stage 0_preflight
[ -d "$DATA_ROOT" ] || fail "dataset root $DATA_ROOT does not exist"
if [ "$PUSH" = "1" ]; then
  # Asked for AFTER 68 h of training is the wrong time to discover it is missing.
  [ -n "${HF_TOKEN:-}" ] || fail "PUSH=1 but HF_TOKEN is not in the environment. Export it in \
YOUR shell for this invocation; nothing writes it to this box."
  [ -n "$HF_REPO" ] || fail "PUSH=1 needs HF_REPO (a private repo id)"
fi
NPROC=$(nproc)
[ "$WORKERS" -lt "$NPROC" ] || echo "WARNING: WORKERS=$WORKERS on a $NPROC-vCPU box; \
oversubscribed workers contend and can be slower than fewer."
$PY -c "import jax; assert jax.devices(), 'no JAX device'; print('jax devices', jax.devices())" \
  || fail "JAX sees no GPU"
OPENPI_SHA=$(git -C /opt/openpi rev-parse --short HEAD 2>/dev/null || echo unknown)
B26_SHA=$(git -C $B26 rev-parse --short HEAD 2>/dev/null || echo unknown)

# --- 1 norm stats: ONCE, shared by both arms ---------------------------------
# Separate stats per arm would be a difference between the arms that is not the
# treatment. Rung 1 showed the shared features come out byte-identical anyway; this
# makes that true by construction instead of by luck.
stage 1_norm_stats
STATS_MARK="$ASSETS/.shot1_stats_done"
if [ -f "$STATS_MARK" ]; then
  echo "NORM_STATS_REUSED $(cat "$STATS_MARK")"
else
  $PY $B26/scripts/compute_norm_stats_b1k.py \
    --config pi05_b1k_frozen_vlm --dataset-root "$DATA_ROOT" --repo-id "${TASKS[@]}" \
    --assets-base-dir "$ASSETS" --max-frames "$STATS_FRAMES" --num-workers 8 \
    2>&1 | tee "$OUT/norm_stats.log" | grep -E "config=|frames=|NORM_STATS_OK|FAIL|Error"
  grep -q NORM_STATS_OK "$OUT/norm_stats.log" || fail "norm stats did not complete"
  date -u +%FT%TZ > "$STATS_MARK"
fi

# --- 2 the two arms ----------------------------------------------------------
# Re-invocation must cost the REMAINING work. A finished arm is skipped; an
# unfinished one resumes from its last checkpoint.
arm_done () { [ -f "$OUT/$1.done" ]; }

run_arm () {
  local name="$1"; shift
  if arm_done "$name"; then echo "ALREADY_DONE $name ($(cat "$OUT/$name.done"))"; return 0; fi
  local resume=""
  [ -d "$CKPT/pi05_b1k_frozen_vlm/$name" ] && resume="--resume"
  echo "--- ARM $name steps=$STEPS batch=$BATCH workers=$WORKERS ${resume:-fresh} $(date -u +%H:%M:%S)"
  ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits | tr -d ' ')" >> "$OUT/gpu_$name.csv"; sleep 10; done ) &
  local sampler=$!
  # The trainer's output goes through a timestamping pipe. `set -o pipefail` (above)
  # makes the SUBSHELL exit non-zero when the trainer does, so $? on the subshell is
  # the honest answer. Not PIPESTATUS: after a subshell that array holds one element,
  # the subshell's own status, so indexing [0] reads like the inner pipe and is not.
  # Getting this wrong records a failed 68-hour run as a success.
  ( cd $B26 && $PY -u scripts/train_b1k_rooted.py \
      --config pi05_b1k_frozen_vlm --exp-name "$name" \
      --dataset-root $ROOT --repo-id "${TASKS[@]}" --protocol \
      --assets-base-dir $ASSETS --checkpoint-base-dir "$CKPT" \
      --num-train-steps $STEPS --batch-size "$BATCH" --num-workers "$WORKERS" --seed $SEED \
      --save-interval $SAVE_EVERY --log-interval 50 $resume "$@" 2>&1 | eval "$TSTAMP" ) \
    >> "$OUT/train_$name.log"
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  echo "ARM_${name}_RC=$rc steps_logged=$(tr '\r' '\n' < "$OUT/train_$name.log" | grep -c 'Step [0-9]*:')"
  if [ "$rc" -eq 0 ]; then date -u +%FT%TZ > "$OUT/$name.done"; else
    echo "ARM $name did not finish (rc=$rc). Re-invoke this script to resume from its \
last checkpoint; nothing is lost."
    return 1
  fi
}

stage 2_arm_A
run_arm armA || fail "arm A"

stage 3_arm_B
# The treatment, and the ONLY difference from arm A.
run_arm armB --progress-loss-weight "$LAMBDA" || fail "arm B"

# --- 4 what was actually run -------------------------------------------------
# A checkpoint nobody can attribute is not a result. Steps, lambda, seed, the data
# fingerprint and both commits, written next to the logs.
stage 4_manifest
{
  echo "{"
  echo "  \"finished_utc\": \"$(date -u +%FT%TZ)\","
  echo "  \"steps\": $STEPS,"
  echo "  \"lambda\": $LAMBDA,"
  echo "  \"seed\": $SEED,"
  echo "  \"batch_size\": $BATCH,"
  echo "  \"workers\": $WORKERS,"
  echo "  \"tasks\": \"$TASKS_CSV\","
  echo "  \"dataset_root\": \"$DATA_ROOT\","
  echo "  \"slice_manifest_sha\": \"$(sha256sum "$DATA_ROOT/meta/slice_manifest.json" 2>/dev/null | cut -c1-16)\","
  echo "  \"norm_stats_sha\": \"$(find "$ASSETS" -name norm_stats.json -exec sha256sum {} \; 2>/dev/null | cut -c1-16 | head -1)\","
  echo "  \"openpi\": \"$OPENPI_SHA\","
  echo "  \"behavior26\": \"$B26_SHA\""
  echo "}"
} > "$OUT/manifest.json"
cat "$OUT/manifest.json"

# --- 5 get the results off the box -------------------------------------------
stage 5_push
if [ "$PUSH" = "1" ]; then
  # HF_TOKEN comes from the invoking environment and stays there. huggingface_hub
  # takes it as an argument; nothing is written to ~/.cache or to any file here.
  $PY - "$HF_REPO" "$CKPT" <<'PYEOF' || fail "push"
import os, sys
from huggingface_hub import HfApi
repo, ckpt = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=ckpt, repo_id=repo, repo_type="model")
print("PUSHED", repo)
PYEOF
else
  echo "PUSH=0 -- checkpoints stay on container disk at $CKPT. Logs and the manifest \
in $OUT are what to salvage before terminating the pod."
fi
echo "SHOT1_DONE $(date -u +%H:%M:%S)"
