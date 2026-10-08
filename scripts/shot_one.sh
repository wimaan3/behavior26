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
# THE POD CANNOT STOP ITSELF. RunPod's injected per-pod key returns 403 for get/stop/
# terminate on its own pod (verified 2026-09-30), and an account key on the rented box is
# forbidden. So every exit path -- success, failure, NOGO gate, health stop, hours cap --
# writes a TERMINAL line to $RUN/STATUS, and scripts/watchdog.sh, running on the owner's
# laptop, terminates the pod when it sees one. The pod's lifetime must not depend on
# anyone watching: the 2026-09-16 loader sweep was lost, and billed six unattended
# hours, because it did.
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
WORKERS="${WORKERS:-auto}"    # auto = min(24, nproc - 3); rung 4's 8 stalled, see the gate below
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-1}"   # what rung 4 measured with; lets rung_report.stalls() read the run
TASKS_CSV="${TASKS_CSV:-set_up_a_coffee_station_in_your_kitchen,putting_shoes_on_rack}"
STATS_FRAMES="${STATS_FRAMES:-128000}"
CFG_A=pi05_b1k_frozen_vlm
CFG_B=pi05_b1k_frozen_vlm_progress
# WARM START (shot two). INIT_FROM: where BOTH arms' weights come from instead of pi05_base --
# "released" = the challenge's released turning_on_radio checkpoint, or a params directory.
# STATS_FROM: reuse that checkpoint's own norm_stats.json ("released", or a checkpoint dir)
# instead of recomputing; recomputed stats would differ slightly from what it was trained with.
INIT_FROM="${INIT_FROM:-}"
STATS_FROM="${STATS_FROM:-}"
GDRIVE_ID="${GDRIVE_ID:-1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2}"   # the released checkpoint (baselines.html)

# --- safeguards --------------------------------------------------------------
GATE_STEP="${GATE_STEP:-300}"
GATE_MAX_MEAN_S="${GATE_MAX_MEAN_S:-4.5}"   # 3.90 s GPU-bound + 15%, fixed in analysis/stall_gate.py
GATE_ENFORCE="${GATE_ENFORCE:-1}"           # 1 = a NOGO stops the run and the pod
MAX_HOURS="${MAX_HOURS:-75}"                # both arms at 3.9 s/step is ~65 h; per invocation
RESUME_TEST="${RESUME_TEST:-1}"             # 1 = exercise --resume once, right after arm A's first checkpoint
MAX_RETRIES="${MAX_RETRIES:-3}"             # a trainer CRASH resumes from its checkpoint this many times
HEALTH_CHECK_STEP="${HEALTH_CHECK_STEP:-1000}"  # analysis/health_gate.py, calibrated on the rung logs
STALL_MINUTES="${STALL_MINUTES:-30}"        # no trainer output for this long = hung: kill the group, auto-resume
CKPT_NEED_GB="${CKPT_NEED_GB:-50}"          # peak: arm A final + arm B latest + one in flight, ~16 GB each
VOL_QUOTA_GB="${VOL_QUOTA_GB:-150}"         # the volume's provisioned size; the API reports it, df cannot

# --- where things live -------------------------------------------------------
VOL="${VOL:-/workspace}"                    # the network volume: survives the pod
RUN="${RUN:-$VOL/shot1}"
BASELINE_ROOT="${BASELINE_ROOT:-$VOL/baseline}"   # the released checkpoint, fetched once, kept on the volume
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
declare -A CHUNK=([set_up_a_coffee_station_in_your_kitchen]=chunk-010 [putting_shoes_on_rack]=chunk-022
                  [turning_on_radio]=chunk-000)
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

# --- every exit leaves a TERMINAL status for the laptop watchdog --------------------
finish () {
  local rc=$?
  kill "${SUPERVISOR:-}" 2>/dev/null
  [ -f "$RUN/STATUS" ] || status "EXITED rc=$rc"
  echo "TERMINAL $(date -u +%FT%TZ)" >> "$RUN/STATUS"
  sync
  echo "SHOT1_EXIT rc=$rc after $(( ($(date +%s) - T_START) / 60 )) min; status: $(tr '\n' ' ' < "$RUN/STATUS")"
  echo "The pod is still billing: scripts/watchdog.sh on the laptop terminates it on TERMINAL."
}
trap finish EXIT
# A signal must not read as success: attempt 1, stopped with SIGTERM, recorded "EXITED rc=0".
trap '[ -f "$RUN/STATUS" ] || status "KILLED by signal"; exit 143' TERM INT HUP
rm -f "$RUN/STATUS"

# --- 0 preflight: everything that can fail cheaply, before anything expensive --
stage 0_preflight
[ -n "${RUNPOD_POD_ID:-}" ] || fail "RUNPOD_POD_ID unset -- the watchdog needs to know which pod to terminate"
echo "$RUNPOD_POD_ID" > "$RUN/pod_id"
[ "$(stat -f -c %T "$VOL" 2>/dev/null)" != "overlayfs" ] || fail "$VOL is container disk, not the network volume"
# NOT df: on a RunPod network volume df reports the whole shared cluster (439 TB free was
# observed), so a df check passes on a volume with 11 GB left. du counts real blocks
# against the quota we provisioned. It walks the evaluator env, so it takes minutes, once.
echo "measuring volume usage with du (minutes)..."
USED_GB=$(du -s --block-size=1G "$VOL" 2>/dev/null | cut -f1)
[ -n "$USED_GB" ] || fail "du could not measure $VOL"
FREE_GB=$(( VOL_QUOTA_GB - USED_GB ))
echo "volume $VOL: ${USED_GB} GB used of ${VOL_QUOTA_GB} GB, ${FREE_GB} GB free (need ${CKPT_NEED_GB})"
[ "$FREE_GB" -ge "$CKPT_NEED_GB" ] || fail "only ${FREE_GB} GB free on $VOL (${USED_GB} of ${VOL_QUOTA_GB} used); \
checkpoints need ~${CKPT_NEED_GB}. Grow the volume and set VOL_QUOTA_GB to its new size."
NPROC=$(nproc)
# 24 is the loader-fix prediction; on a smaller box, oversubscribing the vCPUs makes the
# loader slower, not faster, so leave 3 for the trainer process and the supervisor.
if [ "$WORKERS" = "auto" ]; then WORKERS=$(( NPROC - 3 < 24 ? NPROC - 3 : 24 )); fi
echo "vCPUs $NPROC -> $WORKERS dataloader workers"
[ "$WORKERS" -lt "$NPROC" ] || echo "WARNING: WORKERS=$WORKERS on a $NPROC-vCPU box"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)

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

# The released checkpoint, when either INIT_FROM or STATS_FROM asks for it. ~17 GB from
# Google Drive: downloaded to container disk, unpacked onto the volume, fetched only once.
released_dir () { find "$BASELINE_ROOT" -maxdepth 3 -type d -name params -printf '%h\n' 2>/dev/null | head -1; }
if [ "$INIT_FROM" = "released" ] || [ "$STATS_FROM" = "released" ]; then
  if [ -z "$(released_dir)" ]; then
    echo "fetching the released checkpoint to $BASELINE_ROOT (once)..."
    mkdir -p "$BASELINE_ROOT" /opt/baseline_dl
    (cd $OPENPI_ROOT && uv pip install -q gdown 2>&1 | tail -1
     uv run -- gdown "$GDRIVE_ID" -O /opt/baseline_dl/baseline.download 2>&1 | tail -2) || fail "gdown of the released checkpoint"
    case "$(file -b --mime-type /opt/baseline_dl/baseline.download)" in
      application/zip) unzip -q /opt/baseline_dl/baseline.download -d "$BASELINE_ROOT" || fail "unzip released checkpoint" ;;
      application/gzip|application/x-gzip) tar xzf /opt/baseline_dl/baseline.download -C "$BASELINE_ROOT" || fail "untar released checkpoint" ;;
      *) fail "released checkpoint download is $(file -b /opt/baseline_dl/baseline.download | cut -c1-80), not an archive" ;;
    esac
    rm -rf /opt/baseline_dl
  fi
  REL=$(released_dir); [ -n "$REL" ] || fail "no params/ directory under $BASELINE_ROOT after the fetch"
  echo "released checkpoint: $REL ($(du -sh "$REL" | cut -f1))"
  [ "$INIT_FROM" = "released" ] && INIT_FROM="$REL/params"
  [ "$STATS_FROM" = "released" ] && STATS_FROM="$REL"
fi
INIT_ARGS=(); INIT_PARAMS="$INIT_FROM"; HEALTH_MODE=()
if [ -n "$INIT_PARAMS" ]; then
  [ -d "$INIT_PARAMS" ] || fail "INIT_FROM $INIT_PARAMS is not a directory"
  INIT_ARGS=( --init-from "$INIT_PARAMS" )
  HEALTH_MODE=( --warm-start )
  echo "WARM START: both arms initialise from $INIT_PARAMS"
fi
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
# Drop the depth streams the model never reads -- for EVERY task, including ones built on
# an earlier attempt. LeRobot decodes every video feature in info.json for every sample;
# measured 2026-10-01 on 15 real samples: 4.09 s/sample with all six streams, 1.03 s with
# the three RGB ones. That decode cost WAS the training stall.
for T in "${TASKS[@]}"; do
  $PY $B26/scripts/drop_video_streams.py --root "$ROOT/$T" || fail "drop depth streams for $T"
done
# Prove it on the real data: the streams LeRobot will decode are exactly the three the b1k
# robot config maps (openpi src/openpi/configs/robots/b1k.py), and a sample still loads.
(cd $B26 && $PY - "$ROOT" "${TASKS[@]}" <<'PYEOF') || fail "dataset streams check"
import sys
from openpi.training import lerobot_compat as lc
root, tasks = sys.argv[1], sys.argv[2:]
want = {"observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0"}
for t in tasks:
    ds = lc.LeRobotDataset(repo_id=t, root=f"{root}/{t}", video_backend="pyav", tolerance_s=5e-4)
    got = set(ds.meta.video_keys)
    assert got == want, f"{t}: video_keys {sorted(got)} != {sorted(want)}"
    item = ds[len(ds) // 2]
    assert all(k in item for k in want), f"{t}: a decoded sample lacks an RGB stream"
    print(f"STREAMS_OK {t}: {sorted(got)}")
PYEOF

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
if [ ! -f "$ASSETS/.stats_done" ] && [ -n "$STATS_FROM" ]; then
  # The checkpoint's own statistics, for both arms. There must be exactly one such file:
  # two would mean guessing which one the checkpoint was trained with.
  mapfile -t SRC < <(find "$STATS_FROM/assets" -name norm_stats.json 2>/dev/null)
  [ "${#SRC[@]}" -eq 1 ] || fail "expected exactly one norm_stats.json under $STATS_FROM/assets, found ${#SRC[@]}"
  for c in $CFG_A $CFG_B; do
    mkdir -p "$ASSETS/$c/${TASKS[0]}" && cp "${SRC[0]}" "$ASSETS/$c/${TASKS[0]}/norm_stats.json" || fail "install norm stats for $c"
  done
  echo "norm stats reused from ${SRC[0]}" | tee "$RUN/norm_stats.log"
  date -u +%FT%TZ > "$ASSETS/.stats_done"
fi
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
  # Its own process group (exec setsid keeps the pid, so pid = pgid), recorded for
  # kill_trainer: killing the python alone left its dataloader workers holding this
  # function's output pipe -- the 14.5-hour deadlock of 2026-10-01.
  (cd $B26 && echo $BASHPID > "$RUN/trainer.pgid" && exec setsid $PY -u scripts/train_b1k_rooted.py \
      --config "$cfg" --exp-name "$name" \
      --dataset-root "$DATA_ROOT" --repo-id "${TASKS[@]}" --protocol \
      --assets-base-dir $ASSETS --checkpoint-base-dir "$CKPT" \
      --num-train-steps $STEPS --batch-size "$BATCH" --num-workers "$WORKERS" --seed $SEED \
      --save-interval $SAVE_EVERY --keep-period 0 --log-interval $LOG_EVERY \
      --lr-decay-steps $STEPS "${INIT_ARGS[@]}" "$@")
}

# Kill the trainer AND everything it spawned. The pipe it writes to only reaches EOF when
# every holder is gone; a surviving worker means run_arm waits forever.
kill_trainer () {
  local g i; g=$(cat "$RUN/trainer.pgid" 2>/dev/null)
  [ -n "$g" ] || return 0
  kill -TERM -- "-$g" 2>/dev/null
  for i in $(seq 1 30); do pgrep -g "$g" >/dev/null 2>&1 || return 0; sleep 1; done
  kill -KILL -- "-$g" 2>/dev/null
  return 0
}

# A trainer that died on its own (crash, OOM) can leave the same orphans. If the group
# leader is gone but the group is not, for ORPHAN_GRACE_LOOPS supervisor passes, kill it.
reap_orphans () {
  local g; g=$(cat "$RUN/trainer.pgid" 2>/dev/null)
  if [ -n "$g" ] && ! kill -0 "$g" 2>/dev/null && pgrep -g "$g" >/dev/null 2>&1; then
    ORPHAN_SEEN=$(( ${ORPHAN_SEEN:-0} + 1 ))
    if [ "$ORPHAN_SEEN" -ge "${ORPHAN_GRACE_LOOPS:-2}" ]; then
      echo "REAP: trainer $g is gone but its process group lives on, holding the log pipe; killing it"
      kill -KILL -- "-$g" 2>/dev/null; ORPHAN_SEEN=0
    fi
  else
    ORPHAN_SEEN=0
  fi
}

# --- 3 prove both arms before either trains -----------------------------------
stage 3_check_both_arms
trainer armA "${ARM_A[@]}" --check-only > "$RUN/check_armA.log" 2>&1
grep -q CONFIG_OK "$RUN/check_armA.log" || fail "arm A config (see $RUN/check_armA.log)"
trainer armB "${ARM_B[@]}" --check-only > "$RUN/check_armB.log" 2>&1
grep -q CONFIG_OK "$RUN/check_armB.log" || fail "arm B config (see $RUN/check_armB.log)"
grep -E "^(config|progress_head|progress_key|progress_weight|norm_stats loaded)" "$RUN"/check_arm?.log

# --- 3b warm start only: arm B really loads the checkpoint, before arm A's hours ------
# --check-only never loads weights. Arm B takes the released checkpoint PLUS a new head
# (missing_regex); run its real trainer call until step SMOKE_STEPS is logged, into a scratch
# checkpoint dir, and apply the warm-start rule to its opening loss.
stage 3b_smoke_arm_B
SMOKE_STEPS="${SMOKE_STEPS:-25}"
if [ -n "$INIT_PARAMS" ] && [ ! -f "$RUN/smoke_armB.ok" ]; then
  rm -rf /opt/smoke_ckpt; : > "$RUN/smoke_armB.log"
  ( CKPT=/opt/smoke_ckpt; trainer smokeB "${ARM_B[@]}" 2>&1 | eval "$TSTAMP" ) >> "$RUN/smoke_armB.log" &
  SMOKE_PID=$!
  for i in $(seq 1 180); do      # up to 30 min: model build + JIT + the first steps
    sleep 10
    tr '\r' '\n' < "$RUN/smoke_armB.log" | grep -q "Step $SMOKE_STEPS:" && break
    kill -0 "$SMOKE_PID" 2>/dev/null || break
  done
  kill_trainer; wait "$SMOKE_PID" 2>/dev/null; rm -rf /opt/smoke_ckpt
  tr '\r' '\n' < "$RUN/smoke_armB.log" | grep -q "Step $SMOKE_STEPS:" \
    || { tail -5 "$RUN/smoke_armB.log"; fail "arm B smoke never reached step $SMOKE_STEPS (see $RUN/smoke_armB.log)"; }
  (cd $B26 && $PY -m analysis.health_gate "$RUN/smoke_armB.log" --arm B --warm-start \
      --check-step $HEALTH_CHECK_STEP) > "$RUN/smoke_armB.txt" 2>&1
  [ $? -ne 1 ] || fail "arm B smoke: $(cat "$RUN/smoke_armB.txt")"
  tr '\r' '\n' < "$RUN/smoke_armB.log" | grep -o "Step $SMOKE_STEPS:.*" | tail -1
  date -u +%FT%TZ > "$RUN/smoke_armB.ok"
fi

# --- supervisor: the hours cap and the step-GATE_STEP loader gate ------------------
supervise () {
  local deadline=$(( T_START + MAX_HOURS * 3600 ))
  local ckA="$CKPT/$CFG_A/armA"
  while sleep 60; do
    reap_orphans
    # hung: a trainer group is alive but has written nothing for STALL_MINUTES. Kill the
    # group; run_arm sees a crash and resumes from the last checkpoint. (No STATUS: a hang
    # is retried, not a verdict.) Without this, 2026-10-01 idled 14.5 h.
    local g live
    g=$(cat "$RUN/trainer.pgid" 2>/dev/null); live=""
    [ -f "$RUN/train_armA.log" ] && [ ! -f "$RUN/armA.done" ] && live="$RUN/train_armA.log"
    [ -z "$live" ] && [ -f "$RUN/train_armB.log" ] && [ ! -f "$RUN/armB.done" ] && live="$RUN/train_armB.log"
    if [ -n "$g" ] && [ -n "$live" ] && pgrep -g "$g" >/dev/null 2>&1 \
       && [ $(( $(date +%s) - $(stat -c %Y "$live") )) -ge $(( STALL_MINUTES * 60 )) ]; then
      echo "HUNG: no output in $(basename "$live") for ${STALL_MINUTES} min; killing the trainer group to resume $(date -u +%FT%TZ)"
      kill_trainer
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      status "CAP_HIT after ${MAX_HOURS} h"; kill_trainer; return
    fi
    # speed: the step-GATE_STEP loader gate, arm A only
    if [ ! -f "$RUN/gate.txt" ] && [ -f "$RUN/train_armA.log" ]; then
      (cd $B26 && $PY -m analysis.stall_gate "$RUN/train_armA.log" --gate-step $GATE_STEP \
          --max-mean-s $GATE_MAX_MEAN_S) > "$RUN/gate.try" 2>&1
      case $? in
        0) mv "$RUN/gate.try" "$RUN/gate.txt"; cat "$RUN/gate.txt" ;;
        1) mv "$RUN/gate.try" "$RUN/gate.txt"; cat "$RUN/gate.txt"
           if [ "$GATE_ENFORCE" = "1" ]; then
             status "GATE_NOGO $(cat "$RUN/gate.txt")"; kill_trainer; return
           fi ;;
      esac
    fi
    # health: NaN at any time; loss falling (and Branch 0 for arm B) at HEALTH_CHECK_STEP
    local arm
    for arm in A B; do
      [ -f "$RUN/train_arm$arm.log" ] || continue
      [ -f "$RUN/arm$arm.done" ] && continue
      (cd $B26 && $PY -m analysis.health_gate "$RUN/train_arm$arm.log" --arm $arm \
          --check-step $HEALTH_CHECK_STEP "${HEALTH_MODE[@]}") > "$RUN/health.try" 2>&1
      case $? in
        0) [ -f "$RUN/health_arm$arm.txt" ] || { mv "$RUN/health.try" "$RUN/health_arm$arm.txt"; cat "$RUN/health_arm$arm.txt"; } ;;
        1) mv "$RUN/health.try" "$RUN/health_arm$arm.txt"; cat "$RUN/health_arm$arm.txt"
           status "HEALTH_FAIL $(cat "$RUN/health_arm$arm.txt")"; kill_trainer; return ;;
      esac
    done
    # resume: exercise it ONCE, right after arm A's first checkpoint is finalised.
    # orbax writes into a *.orbax-checkpoint-tmp-* dir and renames it when complete;
    # killing mid-save could corrupt the only checkpoint.
    if [ "$RESUME_TEST" = "1" ] && [ ! -f "$RUN/resume_test.killed" ] && [ -d "$ckA/$SAVE_EVERY" ] \
       && ! ls -d "$ckA"/*orbax-checkpoint-tmp* >/dev/null 2>&1; then
      echo "RESUME_TEST: checkpoint $SAVE_EVERY finalised; killing the trainer once $(date -u +%FT%TZ)"
      date -u +%FT%TZ > "$RUN/resume_test.killed"; kill_trainer
    fi
    if [ -f "$RUN/resume_test.killed" ] && [ ! -f "$RUN/resume_test.txt" ] && [ -f "$RUN/train_armA.log" ]; then
      local first
      first=$(tr '\r' '\n' < "$RUN/train_armA.log" | awk '/RESUME_TEST_RESTART/{f=1; next}
              f && match($0, /Step [0-9]+:/) {print substr($0, RSTART+5, RLENGTH-6); exit}')
      if [ -n "$first" ]; then
        if [ "$first" -ge "$SAVE_EVERY" ]; then
          echo "RESUME_TEST PASS: restarted at step $first (checkpoint $SAVE_EVERY)" | tee "$RUN/resume_test.txt"
        else
          echo "RESUME_TEST FAIL: restarted at step $first, not from checkpoint $SAVE_EVERY" | tee "$RUN/resume_test.txt"
          status "RESUME_BROKEN $(cat "$RUN/resume_test.txt")"; kill_trainer; return
        fi
      fi
    fi
  done
}
supervise & SUPERVISOR=$!

# --- 4 the two arms ------------------------------------------------------------
run_arm () {   # name, then the arm's config + extra args
  local name="$1"; shift
  if [ -f "$RUN/$name.done" ]; then echo "ALREADY_DONE $name ($(cat "$RUN/$name.done"))"; return 0; fi
  ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits | tr -d ' ')" >> "$RUN/gpu_$name.csv"; sleep 30; done ) &
  local sampler=$! attempt=0 rc=1 resume
  while :; do
    # Decided per attempt: the first starts fresh; a retry sees the checkpoint dir (openpi
    # starts fresh itself if the dir holds no checkpoint yet).
    resume=""
    [ -d "$CKPT/$1/$name" ] && resume="--resume"
    echo "--- ARM $name $* steps=$STEPS workers=$WORKERS ${resume:-fresh} attempt $attempt $(date -u +%FT%TZ)"
    # Any resume while the resume test is unverified is evidence for it -- including a
    # fresh launch on a NEW pod resuming from the volume's checkpoint. Mark it so the
    # supervisor checks that the first step logged after it is >= the checkpoint step.
    if [ -n "$resume" ] && [ -f "$RUN/resume_test.killed" ] && [ ! -f "$RUN/resume_test.txt" ]; then
      echo "$(date +%s) === RESUME_TEST_RESTART (resuming from checkpoint; verification pending)" >> "$RUN/train_$name.log"
    fi
    # pipefail makes the subshell exit non-zero when the trainer does, so $? is the
    # trainer's own status, not the timestamper's. A failed run must not read as done.
    ( trainer "$name" "$@" $resume 2>&1 | eval "$TSTAMP" ) >> "$RUN/train_$name.log"
    rc=$?
    kill_trainer      # nothing of this attempt may survive into the next (GPU memory, the pipe)
    echo "ARM_${name}_RC=$rc last $(tr '\r' '\n' < "$RUN/train_$name.log" | grep -o 'Step [0-9]*' | tail -1)"
    [ "$rc" -eq 0 ] && break
    # A deliberate stop (cap, gate, health, broken resume) wrote STATUS first: never retry it.
    [ -f "$RUN/STATUS" ] && break
    if [ -f "$RUN/resume_test.killed" ] && [ ! -f "$RUN/resume_test.restarted" ]; then
      touch "$RUN/resume_test.restarted"
      echo "$(date +%s) === RESUME_TEST_RESTART (deliberate kill; resuming from checkpoint)" >> "$RUN/train_$name.log"
      continue
    fi
    attempt=$(( attempt + 1 ))
    [ "$attempt" -le "$MAX_RETRIES" ] || break
    echo "$(date +%s) === AUTO_RESUME attempt $attempt/$MAX_RETRIES after rc=$rc" >> "$RUN/train_$name.log"
    sleep 30
  done
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
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
  "vcpus": $NPROC,
  "gpu": "$GPU_NAME",
  "resume_test": "$(cat "$RUN/resume_test.txt" 2>/dev/null)",
  "health_armA": "$(cat "$RUN/health_armA.txt" 2>/dev/null)",
  "health_armB": "$(cat "$RUN/health_armB.txt" 2>/dev/null)",
  "config_a": "$CFG_A",
  "config_b": "$CFG_B",
  "init_from": "$INIT_PARAMS",
  "stats_from": "$STATS_FROM",
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
