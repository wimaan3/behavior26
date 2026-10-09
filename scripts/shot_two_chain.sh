#!/usr/bin/env bash
#
# SHOT TWO, END TO END ON ONE POD: wait for training, evaluate both arms, compare.
#
# Why: the laptop used to hand each stage to the next (terminate the training pod, create an
# evaluation pod, launch over SSH). With the laptop closed that hand-off never happened. This
# script does it on the pod, so the RESULT is produced with nobody connected.
#
# What a pod still cannot do, by rule and by RunPod: terminate itself (its injected key gets
# 403; an account key on rented hardware is forbidden) and push to GitHub (no token on the
# box). So this ends like every other stage: TERMINAL in $RUN/STATUS for whoever is watching,
# and everything it produced is on the volume.
#
# The plan it follows is fixed in docs/sessionE-2026-10-08-shot2/PLAN.md:
#   1. training (scripts/shot_one.sh, already running) must end DONE
#   2. evaluate instances 10-18 x 4 for both arms
#   3. if BOTH arms have zero successes there, stop -- the fine-tune broke the policy
#   4. otherwise finish all 27 instances x 4 and write the paired comparison
#
set -uo pipefail

RUN="${RUN:-/workspace/chain2}"
TRAIN="${TRAIN:-/workspace/shot2}"
EVAL="${EVAL:-/workspace/eval2}"
B26="${B26:-/opt/behavior26}"
EXPERIMENT="${EXPERIMENT:-configs/experiments/002-radio-ab.yaml}"
FIRST_BLOCK="${FIRST_BLOCK:-10 11 12 13 14 15 16 17 18}"
BLOCK_HOURS="${BLOCK_HOURS:-12}"   # 72 attempts at ~7.5 min + two scene loads is ~10 h; 7 cut arm B off (9 Oct)
FULL_HOURS="${FULL_HOURS:-30}"
POLL="${POLL:-60}"
CFG_A=pi05_b1k_frozen_vlm
CFG_B=pi05_b1k_frozen_vlm_progress

mkdir -p "$RUN" || { echo "cannot write $RUN"; exit 1; }
exec >> "$RUN/chain.log" 2>&1
status () { echo "$1 $(date -u +%FT%TZ)" > "$RUN/STATUS"; echo "STATUS $1"; }
finish () {
  local rc=$?
  kill "${BEAT:-}" 2>/dev/null
  [ -f "$RUN/STATUS" ] || status "EXITED rc=$rc"
  echo "TERMINAL $(date -u +%FT%TZ)" >> "$RUN/STATUS"; sync
}
trap finish EXIT
trap '[ -f "$RUN/STATUS" ] || status "KILLED by signal"; exit 143' TERM INT HUP
rm -f "$RUN/STATUS"
echo "=== chain start $(date -u +%FT%TZ)"

# The watchdog judges liveness by this log's age. Write to it only while a real stage log is
# moving, so a hung trainer or evaluator still reads as silence.
( while sleep "$POLL"; do
    f=$(ls -t "$TRAIN"/train_arm*.log "$EVAL"/evaluator_arm*.log "$EVAL"/eval.log 2>/dev/null | head -1)
    [ -n "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 300 ] && echo "alive: $(basename "$f") $(date -u +%T)"
  done ) &
BEAT=$!

# --- 1 training must end DONE ---------------------------------------------------------
until grep -q TERMINAL "$TRAIN/STATUS" 2>/dev/null; do sleep "$POLL"; done
TRAIN_STATUS=$(head -1 "$TRAIN/STATUS")
case "$TRAIN_STATUS" in
  DONE*) echo "training ended: $TRAIN_STATUS" ;;
  *) status "TRAIN_FAILED: $TRAIN_STATUS"; exit 1 ;;
esac
latest () { ls -d "$1"/[0-9]* 2>/dev/null | sort -V | tail -1; }
CKPT_A=$(latest "$TRAIN/checkpoints/$CFG_A/armA"); CKPT_B=$(latest "$TRAIN/checkpoints/$CFG_B/armB")
[ -d "$CKPT_A/params" ] && [ -d "$CKPT_B/params" ] || { status "NO_CHECKPOINT: A=$CKPT_A B=$CKPT_B"; exit 1; }
echo "checkpoints: $CKPT_A | $CKPT_B"

evaluate () {   # $1 = hours cap, $2 = instance subset ("" = all)
  env EXPERIMENT="$EXPERIMENT" ASSET_ID=turning_on_radio CKPT_A="$CKPT_A" CKPT_B="$CKPT_B" \
      OUT="$EVAL" MAX_HOURS="$1" INSTANCES_OVERRIDE="$2" VOL_QUOTA_GB="${VOL_QUOTA_GB:-250}" bash "$B26/scripts/eval_arms.sh" > "$RUN/eval_launch.out" 2>&1
  head -1 "$EVAL/STATUS" 2>/dev/null
}
successes () {  # $1 = arm dir -> "successes attempts"
  python3 - "$1" <<'PYEOF'
import glob, json, sys
qs = [json.load(open(f))["q_score"]["final"] for f in glob.glob(sys.argv[1] + "/json/*.json")]
print(sum(q >= 1.0 for q in qs), len(qs))
PYEOF
}

# --- 2 first block, both arms ----------------------------------------------------------
echo "=== first block: instances $FIRST_BLOCK $(date -u +%FT%TZ)"
S=$(evaluate "$BLOCK_HOURS" "$FIRST_BLOCK")
case "$S" in DONE*) ;; *) status "EVAL_BLOCK_FAILED: $S"; exit 1 ;; esac
read -r SA NA <<< "$(successes "$EVAL/armA")"; read -r SB NB <<< "$(successes "$EVAL/armB")"
echo "first block: arm A $SA/$NA, arm B $SB/$NB successes"

# --- 3 the one stopping rule: both arms at zero ----------------------------------------
if [ "$SA" -eq 0 ] && [ "$SB" -eq 0 ]; then
  status "STOPPED_NO_SUCCESS: arm A 0/$NA, arm B 0/$NB in the first block (PLAN.md rule)"; exit 0
fi

# --- 4 all 27 instances, then the comparison -------------------------------------------
echo "=== full run $(date -u +%FT%TZ)"
S=$(evaluate "$FULL_HOURS" "")
case "$S" in DONE*) ;; *) status "EVAL_FULL_INCOMPLETE: $S"; exit 1 ;; esac
read -r SA NA <<< "$(successes "$EVAL/armA")"; read -r SB NB <<< "$(successes "$EVAL/armB")"
cp "$EVAL/compare.txt" "$RUN/compare.txt" 2>/dev/null
status "DONE: arm A $SA/$NA, arm B $SB/$NB successes; comparison in $EVAL/compare.txt"
