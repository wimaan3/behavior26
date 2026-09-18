#!/usr/bin/env bash
#
# AB_PROTOCOL section 2 -- the noise floor sigma_w.
#
# 12 TRAINING instances of turning_on_radio, r = 6 repeats each, against the
# RELEASED BASELINE policy. Not the null policy: it scores ~0 everywhere, so its
# within-instance variance is trivially zero and sigma_w would read 0.
#
# One evaluator invocation, so the scene loads ONCE (measured: ~12 min warm) and
# every repeat after the first costs a reset plus an episode:
#   72 rollouts x ~189 s + one load  =~ 4 h  =~ $3 at $0.74/h
# (The protocol budgeted $9 for this before scene reuse was measured.)
#
# The decision it serves is coarse -- is f below 0.10 or above 0.20 -- which is the
# difference between 1 and 3 seeds per instance, and 3x the evaluation budget.
#
set -uo pipefail
TASK="${TASK:-turning_on_radio}"
INSTANCES="${INSTANCES:-12}"
REPEATS="${REPEATS:-6}"
# On the NETWORK VOLUME, one directory per run: the 2026-09-16 loader sweep wrote to
# container disk and was lost with its pod.
OUT="${OUT:-/workspace/noise_floor/$(date -u +%Y%m%dT%H%MZ)}"
SELF_STOP="${SELF_STOP:-1}"
B26=/opt/behavior26
export PATH="${HOME}/.local/bin:${PATH}"
CUR=init
stage () { echo; echo "=== STAGE $1 $(date -u +%H:%M:%S)"; CUR="$1"; }
fail  () { echo "NOISE_FLOOR_FAILED_AT=${CUR}: $*" | tee -a "$OUT/STATUS"; exit 1; }

# The pod stops itself however this ends -- its lifetime must not depend on anyone
# watching it. A volume-attached pod may refuse `stop`; the results are on the volume,
# so terminating it loses nothing.
self_stop () {
  [ "$SELF_STOP" = "1" ] || return 0
  sync; runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"
}
mkdir -p "$OUT" || { echo "cannot write $OUT -- is the volume mounted?"; exit 1; }
[ -z "$(ls -A "$OUT")" ] || { echo "$OUT is not empty; refusing to mix runs"; exit 1; }
exec > >(tee -a "$OUT/noise_floor.log") 2>&1
if [ "$SELF_STOP" = "1" ]; then
  [ -n "${RUNPOD_POD_ID:-}" ] && runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1 \
    || fail "runpodctl cannot see this pod, so it could not stop it. Set SELF_STOP=0 only if \
someone will stop it by hand."
fi
[ "$(stat -f -c %T "$(dirname "$OUT")")" != "overlayfs" ] || fail "$OUT is on container disk, not the volume"
trap 'echo "EXIT $(date -u +%FT%TZ)" >> "$OUT/STATUS"; self_stop' EXIT

stage 0_serve
bash $B26/scripts/install_openpi.sh 2>&1 | tail -3
OPENPI_ROOT=/opt/openpi bash $B26/scripts/apply_openpi_patches.sh --compat-only 2>&1 | tail -2
setsid nohup bash $B26/scripts/serve_baseline.sh > /tmp/serve.out 2>&1 </dev/null &
bash $B26/scripts/wait_for_policy_server.sh || { tail -20 /tmp/serve.out; fail "policy server"; }

stage 1_rollouts
# shellcheck disable=SC1091
source /workspace/env.sh || fail "no /workspace/env.sh -- mount the volume with the omnigibson env"
# Video writing is deliberately OFF: sigma_w comes from the JSON, and 72 full-res
# RGBD rollouts of video is tens of GB of container disk that nothing ever reads.
# first_rollout.sh already lost a run to exactly this default.
python -m omnigibson.eval.eval \
  --task-name "$TASK" --host 127.0.0.1 --port 8000 --mode train \
  --instance-indices $(seq 0 $((INSTANCES - 1))) \
  --num-rollouts "$REPEATS" \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir "$OUT" --headless > "$OUT/eval.log" 2>&1
RC=$?
N=$(find "$OUT/json" -name '*.json' 2>/dev/null | wc -l)
echo "eval rc=$RC rollouts=$N (expected $((INSTANCES * REPEATS)))"
# ALL of them, not merely some: a partial run biases sigma_w toward whichever
# instances happened to finish, and it would read as a real (low) noise floor.
[ "$N" -eq "$((INSTANCES * REPEATS))" ] || { tail -30 "$OUT/eval.log"; \
  fail "got $N of $((INSTANCES * REPEATS)) rollouts -- a partial run biases sigma_w"; }

stage 2_estimate
python $B26/analysis/noise_floor.py "$OUT/json" | tee "$OUT/sigma_w.txt"
echo "NOISE_FLOOR_DONE $(date -u +%H:%M:%S)"
