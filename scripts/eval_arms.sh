#!/usr/bin/env bash
#
# EVALUATE shot one: arm A (control) vs arm B (progress head), PAIRED, on the frozen dev
# subset -- configs/experiments/001-dev-loop.yaml is the single source of truth for the
# tasks, the training instances, the mode and the rollout count.
#
# What can quietly turn into a wrong RESULT here, and what stops it:
#   - an unnormalised policy        -> both checkpoints' EXACT stats file checked first;
#                                      ASSET_ID passed to the server (both tasks' stats
#                                      live under the first task's id)
#   - arm B unservable              -> the full patch set (0002 declares the head), and
#                                      one real inference per checkpoint BEFORE Isaac Sim
#   - arms on different instances   -> one shared evaluator call, instances from the config
#   - a test-set instance           -> any id >= 301 is refused
#   - the previous arm still serving-> servers run in their own process group, killed
#                                      whole, and port 8000 must be silent before the next
#   - money running out midway      -> task-major order: both arms of a task back to back,
#                                      so what finishes is complete PAIRS; done markers
#                                      let a re-invocation resume
#
# Results (JSON) go to the volume. Videos: written full-size to container disk, then a
# compact 640-px copy of each goes to the volume while there is room (VIDEO_KEEP_FREE_GB).
# Every exit writes TERMINAL to $OUT/STATUS; the laptop watchdog terminates the pod.
#
set -uo pipefail

# Overridable so ONE arm can be pointed at another config + checkpoint for a reference run,
# e.g. the released checkpoint: ARMS=A CFG_A=pi05_b1k CKPT_A=/workspace/baseline/<dir>.
CFG_A="${CFG_A:-pi05_b1k_frozen_vlm}"
CFG_B="${CFG_B:-pi05_b1k_frozen_vlm_progress}"
CKPT_A="${CKPT_A:-/workspace/shot1/checkpoints/$CFG_A/armA/9999}"
CKPT_B="${CKPT_B:-/workspace/shot1/checkpoints/$CFG_B/armB/9999}"
# shot one trained both tasks as one dataset; openpi filed the one stats file under the
# first task's id. Serving either task must point there.
ASSET_ID="${ASSET_ID:-set_up_a_coffee_station_in_your_kitchen}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/001-dev-loop.yaml}"
MAX_HOURS="${MAX_HOURS:-20}"
STALL_MINUTES="${STALL_MINUTES:-45}"          # evaluator silent this long = hung
VIDEO_KEEP_FREE_GB="${VIDEO_KEEP_FREE_GB:-10}"
VOL_QUOTA_GB="${VOL_QUOTA_GB:-200}"
MAX_REINVOKE="${MAX_REINVOKE:-2}"             # the simulator crashed mid-unit twice on 3-4 Oct, exiting 0
# Narrow the run to a SUBSET of the frozen config (refused if outside it), e.g.
#   ARMS=B TASKS_OVERRIDE=putting_shoes_on_rack INSTANCES_OVERRIDE="10 11 ... 21"
ARMS="${ARMS:-A B}"
# SUBMISSION=1 is the ONLY path onto the public test instances: public_test mode, the 20
# public indices (ids 301-320), exactly one arm, results under its own directory.
SUBMISSION="${SUBMISSION:-0}"
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"
INSTANCES_OVERRIDE="${INSTANCES_OVERRIDE:-}"

VOL="${VOL:-/workspace}"
if [ "$SUBMISSION" = "1" ]; then OUT="${OUT:-/workspace/submission1}"; fi
OUT="${OUT:-/workspace/eval1}"
SCR=/opt/eval_scratch                          # container disk: full-size videos
B26=/opt/behavior26
OPENPI_ROOT=/opt/openpi
PY=$OPENPI_ROOT/.venv/bin/python
export OPENPI_ROOT PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "$OUT" "$SCR" || { echo "cannot write $OUT -- is the volume mounted?"; exit 1; }
exec > >(tee -a "$OUT/eval.log") 2>&1
T_START=$(date +%s)
CUR=init
stage () { echo; echo "=== STAGE $1 $(date -u +%FT%TZ)"; CUR="$1"; }
status () { echo "$1 $(date -u +%FT%TZ)" > "$OUT/STATUS"; echo "STATUS $1"; }
fail  () { if [ -f "$OUT/STATUS" ]; then echo "then FAILED_AT=${CUR}: $*" >> "$OUT/STATUS"
           else status "FAILED_AT=${CUR}: $*"; fi; exit 1; }

SERVER_PG=""; EVAL_PG=""
kill_group () { local g="$1"; [ -n "$g" ] || return 0
  kill -TERM -- "-$g" 2>/dev/null
  local i; for i in $(seq 1 30); do pgrep -g "$g" >/dev/null 2>&1 || return 0; sleep 1; done
  kill -KILL -- "-$g" 2>/dev/null; return 0; }
finish () {
  local rc=$?
  kill "${SUPERVISOR:-}" 2>/dev/null
  kill_group "$EVAL_PG"; kill_group "$SERVER_PG"
  [ -f "$OUT/STATUS" ] || status "EXITED rc=$rc"
  echo "TERMINAL $(date -u +%FT%TZ)" >> "$OUT/STATUS"
  sync
  echo "EVAL_EXIT rc=$rc after $(( ($(date +%s) - T_START) / 60 )) min: $(tr '\n' ' ' < "$OUT/STATUS")"
}
trap finish EXIT
trap '[ -f "$OUT/STATUS" ] || status "KILLED by signal"; exit 143' TERM INT HUP
rm -f "$OUT/STATUS"

# The frozen experiment, read -- not copied -- so the script cannot drift from it.
read_config () {
  local vals
  vals=$($PY - "$B26/$EXPERIMENT" <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print(" ".join(d["tasks"]))
print(d["mode"])
print(" ".join(str(i) for i in d["instances"]))
print(d["num_rollouts"])
print(d["env_wrapper"])
PYEOF
) || return 1
  read -r -a TASKS <<< "$(sed -n 1p <<< "$vals")"
  MODE=$(sed -n 2p <<< "$vals")
  read -r -a INSTANCES <<< "$(sed -n 3p <<< "$vals")"
  ROLLOUTS=$(sed -n 4p <<< "$vals")
  WRAPPER=$(sed -n 5p <<< "$vals")
}

# --- 0 preflight ---------------------------------------------------------------
stage 0_preflight
[ -n "${RUNPOD_POD_ID:-}" ] || fail "RUNPOD_POD_ID unset -- the watchdog needs to know which pod to stop"
echo "$RUNPOD_POD_ID" > "$OUT/pod_id"
[ "$(stat -f -c %T "$VOL" 2>/dev/null)" != "overlayfs" ] || fail "$VOL is container disk, not the network volume"
[ -f "$VOL/env.sh" ] || fail "no $VOL/env.sh -- the simulator env is not on this volume"
for a in $ARMS; do      # only the arms being run need a checkpoint
  if [ "$a" = A ]; then c=$CKPT_A; else c=$CKPT_B; fi
  [ -d "$c/params" ] || fail "no checkpoint at $c"
  [ -f "$c/assets/$ASSET_ID/norm_stats.json" ] || fail "no $c/assets/$ASSET_ID/norm_stats.json -- \
it would be served UNNORMALISED. Present: $(ls "$c/assets" 2>/dev/null | tr '\n' ' ')"
done
echo "measuring volume usage with du (minutes)..."
USED_GB=$(du -s --block-size=1G "$VOL" 2>/dev/null | cut -f1)
FREE_GB=$(( VOL_QUOTA_GB - USED_GB )); VIDEO_BUDGET_MB=$(( (FREE_GB - VIDEO_KEEP_FREE_GB) * 1024 ))
echo "volume: ${USED_GB} GB used of ${VOL_QUOTA_GB}, ${FREE_GB} GB free; video budget ${VIDEO_BUDGET_MB} MB"

# --- 1 setup -------------------------------------------------------------------
stage 1_setup
[ -d $B26/.git ] || git clone -q https://github.com/wimaan3/behavior26.git $B26 || fail clone
git -C $B26 fetch -q origin main && git -C $B26 checkout -q origin/main || fail "behavior26 checkout"
if [ ! -x "$PY" ]; then
  SKIP_BASELINE=1 OPENPI_ROOT=$OPENPI_ROOT bash $B26/scripts/install_openpi.sh 2>&1 | tail -3
  [ -x "$PY" ] || fail "openpi install"
fi
# The FULL set: patch 0002 declares the progress-head config arm B was trained under.
OPENPI_ROOT=$OPENPI_ROOT bash $B26/scripts/apply_openpi_patches.sh 2>&1 | tail -2
read_config || fail "cannot read $EXPERIMENT"
subset_of () {   # $1 = name, $2 = requested (space list), rest = allowed
  local name="$1" req="$2"; shift 2; local x
  for x in $req; do printf '%s\n' "$@" | grep -qx "$x" || fail "$name $x is not in the frozen $EXPERIMENT"; done
}
if [ -n "$TASKS_OVERRIDE" ]; then
  subset_of task "${TASKS_OVERRIDE//,/ }" "${TASKS[@]}"; read -r -a TASKS <<< "${TASKS_OVERRIDE//,/ }"
fi
if [ -n "$INSTANCES_OVERRIDE" ]; then
  subset_of instance "$INSTANCES_OVERRIDE" "${INSTANCES[@]}"; read -r -a INSTANCES <<< "$INSTANCES_OVERRIDE"
fi
for a in $ARMS; do case "$a" in A|B) ;; *) fail "ARMS may only contain A and B" ;; esac; done
ID_OFFSET=0
if [ "$SUBMISSION" = "1" ]; then
  [ "$(wc -w <<< "$ARMS")" -eq 1 ] || fail "SUBMISSION=1 needs exactly one arm (ARMS=A or ARMS=B)"
  MODE=public_test; read -r -a INSTANCES <<< "$(seq 0 19 | tr '\n' ' ')"
  ID_OFFSET=301       # --instance-indices are 0..19; the JSON carries the resolved id 301..320
  echo "SUBMISSION MODE: arm $ARMS, public test instances 301-320"
  mkdir -p "$OUT/submission_files"
  cp "$VOL/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml" "$OUT/submission_files/" 2>/dev/null
  cp -r "$VOL/BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers" "$OUT/submission_files/" 2>/dev/null
  git -C "$VOL/BEHAVIOR-1K" describe --tags --always > "$OUT/submission_files/BEHAVIOR-1K.version" 2>/dev/null
else
  for i in "${INSTANCES[@]}"; do
    [ "$i" -lt 301 ] || fail "instance $i is a TEST instance (>= 301); refusing -- that is tuning on the leaderboard"
  done
  [ "$MODE" = "train" ] || fail "mode $MODE: this A/B runs on training instances only"
fi
EXPECTED=$(( ${#INSTANCES[@]} * ROLLOUTS ))
echo "tasks ${TASKS[*]} | mode $MODE | ${#INSTANCES[@]} instances (${INSTANCES[0]}..${INSTANCES[-1]}) x $ROLLOUTS = $EXPECTED rollouts per arm per task"
FFMPEG=$(command -v ffmpeg || $PY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)
echo "ffmpeg: ${FFMPEG:-none -- full-size videos will be copied while the budget allows}"

# --- 2 smoke: each checkpoint answers one real inference, before Isaac Sim -------------
stage 2_smoke
# The server takes each task's prompt from TASK_REGISTRY (patch 0003 adds ours). The arms
# were trained with prompt_from_task=True, i.e. on the task NAME from meta/tasks.parquet;
# a different prompt would not crash, it would quietly handicap both arms.
$PY - "${TASKS[@]}" <<'PYEOF' || fail "served prompts do not match the training prompts"
import sys
from openpi.configs.tasks import TASK_REGISTRY
for t in sys.argv[1:]:
    got = TASK_REGISTRY.get("b1k", {}).get(t)
    assert got == t, f"{t}: registry prompt {got!r}, trained on {t!r}"
print("PROMPTS_OK", sys.argv[1:])
PYEOF
for spec in "A|$CFG_A|$CKPT_A" "B|$CFG_B|$CKPT_B"; do
  IFS='|' read -r arm cfg ckpt <<< "$spec"
  case " $ARMS " in *" $arm "*) ;; *) continue ;; esac
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY - "$cfg" "$ckpt" "$ASSET_ID" <<'PYEOF' || fail "arm $arm cannot serve"
import dataclasses, sys
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
name, ckpt, asset_id = sys.argv[1:4]
cfg = _config.get_config(name)
# exactly what serve_b1k.py does before create_trained_policy
cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=asset_id, robot_config_name="b1k/R1Pro"))
policy = _policy_config.create_trained_policy(cfg, ckpt, default_prompt="smoke test")
# Build the input from the SAME robot config the policy's B1KInputs uses. openpi's
# make_b1k_example is stale: a 23-number state, while B1KInputs indexes the full proprio
# vector at the config's positions (53+), and cameras are keyed by the config's names.
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
rc = next(t.robot_config for t in data_config.data_transforms.inputs if getattr(t, "robot_config", None) is not None)
n = 1 + max(int(np.max(p.indices)) for p in rc.proprio)
obs = {"observation/state": np.zeros(n, np.float32), "prompt": "smoke test"}
for cam in rc.observations:
    obs[f"observation/{cam}"] = np.zeros((240, 240, 3), np.uint8)
out = policy.infer(obs)
a = np.asarray(out["actions"])
assert a.size and np.isfinite(a).all(), f"non-finite actions {a.shape}"
print(f"SMOKE_OK {name}: actions {a.shape}, |a| max {np.abs(a).max():.3f}")
PYEOF
done

# Finished attempts go to the volume as they appear, not only when the evaluator exits: a
# pod lost mid-unit (host failure, empty balance) would otherwise take every result with it.
sync_results () {
  local d
  for d in "$SCR"/arm*/*/json; do
    [ -d "$d" ] || continue
    local arm; arm=$(basename "$(dirname "$(dirname "$d")")")
    mkdir -p "$OUT/$arm/json" && cp -n "$d"/*.json "$OUT/$arm/json/" 2>/dev/null
  done
  return 0
}

# --- supervisor: hours cap, and a hung evaluator --------------------------------------
supervise () {
  local deadline=$(( T_START + MAX_HOURS * 3600 ))
  while sleep 60; do
    sync_results
    if [ "$(date +%s)" -ge "$deadline" ]; then
      status "CAP_HIT after ${MAX_HOURS} h"; kill_group "$(cat "$OUT/eval.pg" 2>/dev/null)"; return
    fi
    local g log; g=$(cat "$OUT/eval.pg" 2>/dev/null); log=$(cat "$OUT/eval.current" 2>/dev/null)
    if [ -n "$g" ] && [ -f "$log" ] && pgrep -g "$g" >/dev/null 2>&1 \
       && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -ge $(( STALL_MINUTES * 60 )) ]; then
      echo "HUNG: evaluator silent for ${STALL_MINUTES} min; killing it $(date -u +%FT%TZ)"
      kill_group "$g"
    fi
  done
}
supervise & SUPERVISOR=$!

# --- 3 evaluation: task-major, both arms back to back ----------------------------------
stage 3_eval
compact_videos () {   # $1 = source dir, $2 = destination dir on the volume
  mkdir -p "$2"; local v out mb
  for v in $(find "$1" -name '*.mp4' | sort); do
    [ "$VIDEO_BUDGET_MB" -gt 0 ] || { echo "video budget spent; leaving $(basename "$v") on container disk"; continue; }
    out="$2/$(basename "$v")"
    if [ -n "$FFMPEG" ]; then
      "$FFMPEG" -nostdin -loglevel error -y -i "$v" -vf "scale=640:-2" -c:v libx264 -preset veryfast \
        -crf 30 -an "$out" || cp "$v" "$out"
    else
      cp "$v" "$out"
    fi
    mb=$(( $(stat -c %s "$out") / 1048576 + 1 )); VIDEO_BUDGET_MB=$(( VIDEO_BUDGET_MB - mb ))
  done
}
# Instances of $INSTANCES with fewer than ROLLOUTS results in $2 (default 1 rollout each).
missing_instances () {   # $1 = task, $2 = json dir
  local i n
  for i in "${INSTANCES[@]}"; do
    n=$(ls "$2/${1}_$(( i + ${ID_OFFSET:-0} ))"_*.json 2>/dev/null | wc -l)
    [ "$n" -ge "${ROLLOUTS:-1}" ] || echo "$i"
  done
}
FAILED_UNITS=0
for TASK in "${TASKS[@]}"; do
  for ARM in $ARMS; do
    if [ "$ARM" = A ]; then CFG=$CFG_A; CKPT=$CKPT_A; else CFG=$CFG_B; CKPT=$CKPT_B; fi
    UNIT="$OUT/arm$ARM/$TASK"; JSON="$OUT/arm$ARM/json"; mkdir -p "$JSON"
    attempt=0
    while :; do
      # Results already on the volume are kept: a crashed or capped unit resumes with only
      # the instances still missing, at the cost of one scene reload.
      read -r -a TODO <<< "$(missing_instances "$TASK" "$JSON" | tr '\n' ' ')"
      if [ "${#TODO[@]}" -eq 0 ]; then date -u +%FT%TZ > "$UNIT.done"; echo "UNIT arm $ARM $TASK complete"; break; fi
      if [ "$attempt" -gt "$MAX_REINVOKE" ]; then
        FAILED_UNITS=$(( FAILED_UNITS + 1 )); echo "UNIT arm $ARM $TASK INCOMPLETE after $attempt attempts: missing ${TODO[*]}"; break
      fi
      echo "--- UNIT arm $ARM ($CFG) on $TASK, attempt $attempt, instances ${TODO[*]} $(date -u +%FT%TZ)"
      rm -rf "$SCR/arm$ARM/$TASK"

      # the previous arm's server must be gone, or this unit would talk to it
      if curl -sf -m 3 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
        fail "port 8000 still answering before arm $ARM $TASK -- the previous server is still serving"
      fi
      setsid nohup env CONFIG="$CFG" CKPT="$CKPT" TASK="$TASK" ASSET_ID="$ASSET_ID" PORT=8000 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
        bash $B26/scripts/serve_baseline.sh > "$OUT/serve_arm${ARM}_${TASK}.log" 2>&1 < /dev/null &
      SERVER_PG=$!
      if ! bash $B26/scripts/wait_for_policy_server.sh; then
        tail -20 "$OUT/serve_arm${ARM}_${TASK}.log"; kill_group "$SERVER_PG"; SERVER_PG=""
        attempt=$(( attempt + 1 )); continue
      fi

      LOG="$OUT/evaluator_arm${ARM}_${TASK}.log"; echo "$LOG" > "$OUT/eval.current"
      echo "=== attempt $attempt $(date -u +%FT%TZ): instances ${TODO[*]}" >> "$LOG"
      # bash -c's first extra argument becomes $0 and is NOT in "$@"; the env path travels in
      # an environment variable so every evaluator argument reaches "$@" intact. (A `shift`
      # here once discarded --task-name and failed all four units.)
      VOLENV="$VOL/env.sh" setsid bash -c 'source "$VOLENV" && exec python -m omnigibson.eval.eval "$@"' evaluator \
        --task-name "$TASK" --host 127.0.0.1 --port 8000 --mode "$MODE" \
        --instance-indices "${TODO[@]}" --num-rollouts "$ROLLOUTS" \
        --env-wrapper "$WRAPPER" --output-dir "$SCR/arm$ARM/$TASK" \
        --write-video --headless >> "$LOG" 2>&1 < /dev/null &
      EVAL_PG=$!; echo "$EVAL_PG" > "$OUT/eval.pg"
      wait "$EVAL_PG"; rc=$?
      rm -f "$OUT/eval.pg"; EVAL_PG=""
      kill_group "$SERVER_PG"; SERVER_PG=""

      cp "$SCR/arm$ARM/$TASK"/json/*.json "$JSON/" 2>/dev/null
      echo "UNIT arm $ARM $TASK attempt $attempt: evaluator rc=$rc; still missing: $(missing_instances "$TASK" "$JSON" | tr '\n' ' ')"
      compact_videos "$SCR/arm$ARM/$TASK" "$OUT/arm$ARM/videos/$TASK"
      if [ -f "$OUT/STATUS" ]; then exit 1; fi              # the cap fired: stop here
      attempt=$(( attempt + 1 ))
    done
  done
done

# --- 4 the paired comparison -----------------------------------------------------------
stage 4_compare
if [ "$SUBMISSION" = "1" ]; then echo "submission run: no A/B comparison"; else
(cd $B26 && $PY -m analysis.compare "$OUT/armA" "$OUT/armB" --per-instance --csv "$OUT/paired.csv") \
  | tee "$OUT/compare.txt" || echo "compare.py failed (see above)"
fi
if [ "$FAILED_UNITS" -gt 0 ]; then status "PARTIAL: $FAILED_UNITS unit(s) incomplete; re-invoke to finish"
else status "DONE"; fi
