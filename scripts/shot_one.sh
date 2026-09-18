#!/usr/bin/env bash
#
# SHOT ONE -- the A/B the project exists to run.
#
#   arm A  pi05_b1k_frozen_vlm                                   (control)
#   arm B  pi05_b1k_frozen_vlm_progress + progress labels at LAMBDA (treatment)
#
# The arms must differ in the progress head and in NOTHING else. Enforced by
# construction: ONE trainer call (run_arm) fixes data, steps, seed, batch, workers,
# assets and --protocol; the arms differ only in the config name and the two
# arguments that switch the head on. Norm stats are computed once and the SAME file
# is given to both (rung 1: the two configs' stats have identical keys and values).
# Both arms' complete configs are proven with --check-only BEFORE either trains, so a
# mistake in arm B cannot surface ~32 h into arm A.
#
# COST. Rung 4: 3.90 s/step GPU-bound, but 8.11 s/step mean at 8 workers (a prefetch
# stall every 24 steps). WORKERS=24 is predicted to remove it; the step-GATE_STEP gate
# measures that on arm A's own opening steps and stops the pod if it did not.
#
# NOTHING IMPORTANT LIVES ON THE POD. Logs, manifest, gate verdict, norm stats and
# checkpoints go to the network volume ($VOL). openpi and the merged data are rebuilt
# on container disk when missing (a new pod after an interruption), and the data is
# checked against the fingerprint the run started with.
#
# THE POD STOPS ITSELF: on success, on failure, on a NOGO gate, and at MAX_HOURS. Its
# lifetime must not depend on anyone watching it -- the 2026-09-16 loader sweep was
# lost, and billed six unattended hours, because it did.
#
# RESUMING: re-invoke on any pod with the volume. A finished arm is skipped; an
# unfinished one continues from its last checkpoint on the volume.
#
# CREDENTIALS: none. Nothing here logs in or writes a token. Checkpoints stay on the
# volume; there is no push.
#
set -uo pipefail

# --- what to run -------------------------------------------------------------
STEPS="${STEPS:-30000}"
# Rung 3: at lambda 0.1 the progress term is ~14-15% of the update, so ~0.14-0.15
# buys 20%. A parameter, not a literal -- docs/sessionB-2026-09-16-rung2-5/RUNG2-5.md.
LAMBDA="${LAMBDA:-0.15}"
SEED="${SEED:-42}"            # TrainConfig's own default; both arms share it
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-24}"      # rung 4's 8 stalled; see the gate below
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-1}"   # what rung 4 measured with; lets rung_report.stalls() read the run
TASKS_CSV="${TASKS_CSV:-set_up_a_coffee_station_in_your_kitchen,putting_shoes_on_rack}"
STATS_FRAMES="${STATS_FRAMES:-128000}"
CFG_A=pi05_b1k_frozen_vlm
CFG_B=pi05_b1k_frozen_vlm_progress

# --- safeguards --------------------------------------------------------------
GATE_STEP="${GATE_STEP:-300}"
GATE_MAX_MEAN_S="${GATE_MAX_MEAN_S:-4.5}"   # 3.90 s GPU-bound + 15%, fixed in analysis/stall_gate.py
GATE_ENFORCE="${GATE_ENFORCE:-1}"           # 1 = a NOGO stops the run and the pod
MAX_HOURS="${MAX_HOURS:-75}"                # both arms at 3.9 s/step is ~65 h; per invocation
SELF_STOP="${SELF_STOP:-1}"                 # 1 = the pod stops itself when this script ends
CKPT_NEED_GB="${CKPT_NEED_GB:-45}"          # two arms' final checkpoints plus one in flight

# --- where things live -------------------------------------------------------
VOL="${VOL:-/workspace}"                    # the network volume: survives the pod
RUN="${RUN:-$VOL/shot1}"
CKPT="$RUN/checkpoints"
ASSETS="$RUN/assets"
ROOT=/opt/merged                            # rebuilt per pod, fingerprint-checked
B26=/opt/behavior26
OPENPI_ROOT=/opt/openpi
PY=$OPENPI_ROOT/.venv/bin/python
export OPENPI_ROOT PATH="${HOME}/.local/bin:${PATH}" HF_HUB_DISABLE_PROGRESS_BARS=1 GIT_LFS_SKIP_SMUDGE=1
# Training, not serving: preallocate. (Serving is the opposite -- see serve_baseline.sh.)
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Proven bit-identical to full decode on 640 seeded frames (rung 1), ~100x faster.
export NORM_STATS_NO_DECODE=1

read -r -a TASKS <<< "${TASKS_CSV//,/ }"
declare -A CHUNK=([set_up_a_coffee_station_in_your_kitchen]=chunk-010 [putting_shoes_on_rack]=chunk-022)
if [ "${#TASKS[@]}" -eq 1 ]; then DATA_ROOT="$ROOT/${TASKS[0]}"; else DATA_ROOT="$ROOT"; fi

mkdir -p "$RUN" "$CKPT" "$ASSETS" || { echo "cannot write to $RUN -- is the volume mounted at $VOL?"; exit 1; }
LOG="$RUN/shot1.log"
exec > >(tee -a "$LOG") 2>&1
T_START=$(date +%s)
CUR=init
stage () { echo; echo "=== STAGE $1 $(date -u +%FT%TZ)"; CUR="$1"; }
status () { echo "$1 $(date -u +%FT%TZ)" > "$RUN/STATUS"; echo "STATUS $1"; }
# The supervisor records WHY it stopped training (CAP_HIT, GATE_NOGO) before killing
# the trainer; the failure that follows must not overwrite that reason.
fail  () { if [ -f "$RUN/STATUS" ]; then echo "then FAILED_AT=${CUR}: $*" >> "$RUN/STATUS"
           else status "FAILED_AT=${CUR}: $*"; fi; exit 1; }
TSTAMP='while IFS= read -r l; do printf "%s %s\n" "$(date +%s.%N)" "$l"; done'

# --- the pod stops itself, whatever happens -----------------------------------
self_stop () {
  [ "$SELF_STOP" = "1" ] || { echo "SELF_STOP=0: leaving the pod running"; return; }
  sync
  echo "stopping pod ${RUNPOD_POD_ID} $(date -u +%FT%TZ)"
  # A pod with a network volume may refuse `stop`; everything that matters is on the
  # volume, so fall back to terminating it rather than leave it billing.
  runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"
}
finish () {
  local rc=$?
  kill "${SUPERVISOR:-}" 2>/dev/null
  [ -f "$RUN/STATUS" ] || status "EXITED rc=$rc"
  echo "SHOT1_EXIT rc=$rc after $(( ($(date +%s) - T_START) / 60 )) min; status: $(cat "$RUN/STATUS")"
  self_stop
}
trap finish EXIT
rm -f "$RUN/STATUS"

# --- 0 preflight: everything that can fail cheaply, before anything expensive --
stage 0_preflight
if [ "$SELF_STOP" = "1" ]; then
  [ -n "${RUNPOD_POD_ID:-}" ] || { SELF_STOP=0; fail "RUNPOD_POD_ID unset -- cannot stop this pod when done. \
Run on a RunPod pod, or set SELF_STOP=0 knowingly."; }
  runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1 || { SELF_STOP=0; fail "runpodctl cannot see \
this pod, so it could not stop it either. Set SELF_STOP=0 only if someone will stop it by hand."; }
fi
[ "$(stat -f -c %T "$VOL" 2>/dev/null)" != "overlayfs" ] || fail "$VOL is container disk, not the network volume"
FREE_GB=$(df -BG --output=avail "$VOL" | tail -1 | tr -dc 0-9)
echo "volume $VOL free ${FREE_GB} GB (need ${CKPT_NEED_GB})"
[ "$FREE_GB" -ge "$CKPT_NEED_GB" ] || fail "only ${FREE_GB} GB free on $VOL; checkpoints need ~${CKPT_NEED_GB}"
NPROC=$(nproc)
[ "$WORKERS" -lt "$NPROC" ] || echo "WARNING: WORKERS=$WORKERS on a $NPROC-vCPU box"

stage 0_setup
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d $B26/.git ] || git clone -q https://github.com/wimaan3/behavior26.git $B26 || fail clone
git -C $B26 fetch -q origin main && git -C $B26 checkout -q origin/main || fail "behavior26 checkout"
if [ ! -x "$PY" ]; then
  [ -d $OPENPI_ROOT/.git ] || git clone -q --filter=blob:none https://github.com/wensi-ai/openpi.git $OPENPI_ROOT || fail openpi_clone
  git -C $OPENPI_ROOT checkout -q 0cc8e355f7bac0976db1cc3139b1ff0379feea60 || fail openpi_checkout
  (cd $OPENPI_ROOT && uv sync 2>&1 | tail -1) || fail uv_sync
  (cd $OPENPI_ROOT && uv pip install -q av msgpack-numpy 2>&1 | tail -1)
  bash $B26/scripts/apply_openpi_patches.sh 2>&1 | tail -1
fi
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu'; print('jax', jax.__version__, jax.devices())" \
  || fail "JAX sees no GPU"
OPENPI_SHA=$(git -C $OPENPI_ROOT rev-parse --short HEAD)
B26_SHA=$(git -C $B26 rev-parse --short HEAD)

# --- 1 data: rebuilt on each pod, identical every time -------------------------
stage 1_data
for T in "${TASKS[@]}"; do
  [ -f "$ROOT/$T/meta/progress_filter.json" ] && continue
  C=${CHUNK[$T]:-}; [ -n "$C" ] || fail "no chunk known for task $T"
  # The exact form rung 2-5 used successfully.
  $OPENPI_ROOT/.venv/bin/hf download behavior-1k/2026-challenge-demos --repo-type dataset \
      --local-dir /opt/behavior-data \
      --include "data/${C}/**" --include "meta/episodes/${C}/**" --include "videos/*/${C}/**" \
      --include "meta/info.json" --include "meta/stats.json" --include "meta/tasks.parquet" \
      --include "meta/tasks.jsonl" >/dev/null 2>&1 || fail "download $C"
  $PY $B26/scripts/slice_task_dataset.py --source /opt/behavior-data --task "$T" --out /opt/tasks \
      --overwrite 2>&1 | tail -1 || fail "slice $T"
  rm -rf "$ROOT/$T"
  $PY $B26/scripts/merge_progress_labels.py --dataset-root /opt/tasks/$T \
      --labels $B26/labels/$T/labels.parquet --out-root "$ROOT/$T" \
      --join-on global_episode_index frame_index --labels-join-on episode_index frame_index \
      --drop-unlabelled 2>&1 | grep -E "dropped|compacted|FAIL" | tail -3
  [ -f "$ROOT/$T/meta/progress_filter.json" ] || fail "merge $T"
done
# The fingerprint of what was kept. A resumed run on a new pod must train on exactly
# the data the checkpoint was trained on.
FP=$(for T in "${TASKS[@]}"; do sha256sum "$ROOT/$T/meta/progress_filter.json" | cut -c1-16; done | tr '\n' ' ')
if [ -f "$RUN/data_fingerprint" ]; then
  [ "$(cat "$RUN/data_fingerprint")" = "$FP" ] || fail "rebuilt data differs from what this run \
started on ($(cat "$RUN/data_fingerprint") vs $FP) -- resuming would change the training set mid-run"
else
  echo "$FP" > "$RUN/data_fingerprint"
fi
echo "data fingerprint $FP"

# --- 2 norm stats: ONCE, the same file for both arms ---------------------------
stage 2_norm_stats
if [ ! -f "$ASSETS/.stats_done" ]; then
  $PY $B26/scripts/compute_norm_stats_b1k.py --config-name $CFG_A \
      --dataset-root "$DATA_ROOT" --repo-id "${TASKS[@]}" --assets-base-dir "$ASSETS" \
      --max-frames "$STATS_FRAMES" --num-workers 8 > "$RUN/norm_stats.log" 2>&1
  grep -q NORM_STATS_OK "$RUN/norm_stats.log" || fail "norm stats (see $RUN/norm_stats.log)"
  # Arm B reads the SAME statistics: identical keys and values in rung 1, so one file
  # copied is identical by construction rather than by recomputation.
  (cd "$ASSETS/$CFG_A" && find . -name norm_stats.json) | while read -r f; do
    mkdir -p "$ASSETS/$CFG_B/$(dirname "$f")"; cp "$ASSETS/$CFG_A/$f" "$ASSETS/$CFG_B/$f"
  done
  date -u +%FT%TZ > "$ASSETS/.stats_done"
fi
[ "$(find "$ASSETS/$CFG_A" -name norm_stats.json -exec sha256sum {} \; | cut -c1-64)" = \
  "$(find "$ASSETS/$CFG_B" -name norm_stats.json -exec sha256sum {} \; | cut -c1-64)" ] \
  || fail "the two arms' norm stats differ"
STATS_SHA=$(find "$ASSETS/$CFG_A" -name norm_stats.json -exec sha256sum {} \; | cut -c1-16)

# --- the one trainer call ---------------------------------------------------------
# Arms differ only in: config name, and (arm B) --progress-key + --progress-loss-weight.
ARM_A=( "$CFG_A" )
ARM_B=( "$CFG_B" --progress-key progress --progress-loss-weight "$LAMBDA" )

trainer () {   # exp-name config [extra...]; extra args follow the shared ones
  local name="$1" cfg="$2"; shift 2
  (cd $B26 && $PY -u scripts/train_b1k_rooted.py \
      --config "$cfg" --exp-name "$name" \
      --dataset-root $ROOT --repo-id "${TASKS[@]}" --protocol \
      --assets-base-dir $ASSETS --checkpoint-base-dir "$CKPT" \
      --num-train-steps $STEPS --batch-size "$BATCH" --num-workers "$WORKERS" --seed $SEED \
      --save-interval $SAVE_EVERY --keep-period 0 --log-interval $LOG_EVERY "$@")
}

# --- 3 prove both arms before either trains -----------------------------------
stage 3_check_both_arms
trainer armA "${ARM_A[@]}" --check-only > "$RUN/check_armA.log" 2>&1
grep -q CONFIG_OK "$RUN/check_armA.log" || fail "arm A config (see $RUN/check_armA.log)"
trainer armB "${ARM_B[@]}" --check-only > "$RUN/check_armB.log" 2>&1
grep -q CONFIG_OK "$RUN/check_armB.log" || fail "arm B config (see $RUN/check_armB.log)"
grep -E "^(config|progress_head|progress_key|progress_weight|norm_stats loaded)" "$RUN"/check_arm?.log

# --- supervisor: the hours cap and the step-GATE_STEP loader gate ------------------
supervise () {
  local deadline=$(( T_START + MAX_HOURS * 3600 ))
  while sleep 60; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      status "CAP_HIT after ${MAX_HOURS} h"; pkill -f train_b1k_rooted.py; return
    fi
    if [ ! -f "$RUN/gate.txt" ] && [ -f "$RUN/train_armA.log" ]; then
      (cd $B26 && $PY -m analysis.stall_gate "$RUN/train_armA.log" --gate-step $GATE_STEP \
          --max-mean-s $GATE_MAX_MEAN_S) > "$RUN/gate.try" 2>&1
      case $? in
        0) mv "$RUN/gate.try" "$RUN/gate.txt"; echo "$(cat "$RUN/gate.txt")" ;;
        1) mv "$RUN/gate.try" "$RUN/gate.txt"; echo "$(cat "$RUN/gate.txt")"
           if [ "$GATE_ENFORCE" = "1" ]; then
             status "GATE_NOGO $(cat "$RUN/gate.txt")"; pkill -f train_b1k_rooted.py; return
           fi ;;
      esac
    fi
  done
}
supervise & SUPERVISOR=$!

# --- 4 the two arms ------------------------------------------------------------
run_arm () {   # name, then the arm's config + extra args
  local name="$1"; shift
  if [ -f "$RUN/$name.done" ]; then echo "ALREADY_DONE $name ($(cat "$RUN/$name.done"))"; return 0; fi
  local resume=""
  [ -d "$CKPT/$1/$name" ] && resume="--resume"
  echo "--- ARM $name $* steps=$STEPS workers=$WORKERS ${resume:-fresh} $(date -u +%FT%TZ)"
  ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits | tr -d ' ')" >> "$RUN/gpu_$name.csv"; sleep 30; done ) &
  local sampler=$!
  # pipefail makes the subshell exit non-zero when the trainer does, so $? is the
  # trainer's own status, not the timestamper's. A failed run must not read as done.
  ( trainer "$name" "$@" $resume 2>&1 | eval "$TSTAMP" ) >> "$RUN/train_$name.log"
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  echo "ARM_${name}_RC=$rc last $(tr '\r' '\n' < "$RUN/train_$name.log" | grep -o 'Step [0-9]*' | tail -1)"
  [ "$rc" -eq 0 ] || return 1
  date -u +%FT%TZ > "$RUN/$name.done"
}

stage 4_arm_A
run_arm armA "${ARM_A[@]}" || fail "arm A did not finish -- re-invoke to resume"
stage 5_arm_B
run_arm armB "${ARM_B[@]}" || fail "arm B did not finish -- re-invoke to resume"

# --- 6 what was actually run ---------------------------------------------------
stage 6_manifest
cat > "$RUN/manifest.json" <<EOF
{
  "finished_utc": "$(date -u +%FT%TZ)",
  "steps": $STEPS,
  "lambda": $LAMBDA,
  "seed": $SEED,
  "batch_size": $BATCH,
  "workers": $WORKERS,
  "config_a": "$CFG_A",
  "config_b": "$CFG_B",
  "tasks": "$TASKS_CSV",
  "data_fingerprint": "$FP",
  "norm_stats_sha": "$STATS_SHA",
  "gate": "$(cat "$RUN/gate.txt" 2>/dev/null)",
  "openpi": "$OPENPI_SHA",
  "behavior26": "$B26_SHA",
  "checkpoints": "$CKPT"
}
EOF
cat "$RUN/manifest.json"
status "DONE"
