#!/usr/bin/env bash
#
# Runs on the OWNER'S LAPTOP. Terminates the training pod when the run ends.
#
# Why it exists: RunPod's injected per-pod key returns 403 for stop/terminate on its own
# pod (verified 2026-09-30), and an account key on the rented box is forbidden. So the
# pod writes "TERMINAL" to $RUN/STATUS on every exit path (scripts/shot_one.sh) and this
# script -- with the owner's key, on the owner's machine -- terminates it.
#
#   POD_ID=...  SSH_USER=<pod>-<n>  DEADLINE_EPOCH=<unix time>  bash scripts/watchdog.sh [check]
#
#   check   one pass: prove the key works and the pod's STATUS is readable; never terminates
#   (none)  loop every INTERVAL s; terminate on TERMINAL, or at DEADLINE_EPOCH as a backstop
#           in case the pod's own hours cap never fires; exit quietly if the pod is gone
#
# The key is read from KEY_FILE (mode 600) into the environment of each runpodctl call
# only: never into argv (visible in ps) and never into the log.
#
set -uo pipefail
MODE="${1:-run}"
POD_ID="${POD_ID:?set POD_ID}"
SSH_USER="${SSH_USER:?set SSH_USER, e.g. ${POD_ID}-6441226b}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:?set DEADLINE_EPOCH (unix time): the hard backstop}"
RUN="${RUN:-/workspace/shot1}"
KEY_FILE="${KEY_FILE:-$HOME/.runpod/watchdog.key}"
INTERVAL="${INTERVAL:-300}"
# The pod's supervisor kills a hung trainer after 30 min of silence and resumes it. If the
# run has been silent for 3x that, the supervisor itself is gone: stop the billing.
STALL_MIN="${STALL_MIN:-90}"
# Which logs prove the run is alive: training writes train_arm*.log, evaluation
# (scripts/eval_arms.sh) writes evaluator_arm*.log.
LOG_GLOB="${LOG_GLOB:-train_arm*.log}"
MAX_LOOPS="${MAX_LOOPS:-0}"                      # 0 = forever (tests set a bound)
LOG="${LOG:-$HOME/.runpod/watchdog-$POD_ID.log}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$LOG")"
log () { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

[ -f "$KEY_FILE" ] || { echo "no key file $KEY_FILE"; exit 2; }
perm=$(stat -c %a "$KEY_FILE")
[ "$perm" = "600" ] || { echo "refusing: $KEY_FILE is mode $perm; it must be 600 (chmod 600 $KEY_FILE)"; exit 2; }

api () {   # the key exists only in this child's environment
  RUNPOD_API_KEY="$(cat "$KEY_FILE")" runpodctl "$@"
}

# 0 = pod exists, 1 = pod is gone, 2 = could not tell (network, auth)
pod_state () {
  local out
  out=$(api pod get "$POD_ID" 2>&1) && return 0
  echo "$out" | grep -qiE "404|not found" && return 1
  log "pod get failed: $(echo "$out" | head -1 | cut -c1-160)"
  return 2
}

# RunPod's SSH proxy refuses exec and forces a PTY that ECHOES every command and wraps
# prompts in escape codes (seen in `check` against the real pod, 2026-10-01). So: the
# markers are assembled at runtime from two halves (the echoed command text never contains
# them), escape codes are stripped, and only lines tagged "STATUS: " or "LOGAGE <n>" count.
# Echoed text can therefore never read as a verdict, and the age survives prompt noise.
read_status () {
  { printf '%s\n' 'stty -echo 2>/dev/null; PS1=""' \
      "printf '%s%s\\n' __WD START__" \
      "sed 's/^/STATUS: /' $RUN/STATUS 2>/dev/null" \
      "f=\$(ls -t $RUN/$LOG_GLOB 2>/dev/null | head -1); [ -n \"\$f\" ] && echo LOGAGE \$(( \$(date +%s) - \$(stat -c %Y \"\$f\") ))" \
      "printf '%s%s\\n' __WD END__" 'exit'; } \
    | timeout 90 ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 \
        -o ServerAliveInterval=15 -i "$SSH_KEY" "$SSH_USER@ssh.runpod.io" 2>/dev/null \
    | tr -d '\r' \
    | sed -e 's/\x1b\][^\x07]*\x07//g' -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
    | sed -n '/^__WDSTART__$/,/^__WDEND__$/p' \
    | grep -E '^(STATUS: |LOGAGE [0-9]+$)'
}

terminate () {
  log "TERMINATING $POD_ID: $1"
  local out
  if out=$(api pod delete "$POD_ID" 2>&1); then
    log "terminated $POD_ID"
  else
    log "terminate FAILED: $(echo "$out" | head -1 | cut -c1-160) -- retrying next loop"
    return 1
  fi
}

if [ "$MODE" = "check" ]; then
  pod_state; case $? in
    0) log "check: key works, pod $POD_ID visible" ;;
    1) log "check: pod $POD_ID is gone"; exit 1 ;;
    *) log "check: could not reach the API with this key"; exit 1 ;;
  esac
  raw=$(read_status)
  st=$(echo "$raw" | sed -n 's/^STATUS: //p'); age=$(echo "$raw" | sed -n 's/^LOGAGE \([0-9]*\)$/\1/p' | tail -1)
  log "check: STATUS = ${st:-<none yet: run in progress>}; last training output ${age:-?} s ago"
  log "check: deadline $(date -u -d "@$DEADLINE_EPOCH" +%FT%TZ)"
  exit 0
fi

log "watching pod $POD_ID every ${INTERVAL}s; backstop $(date -u -d "@$DEADLINE_EPOCH" +%FT%TZ)"
n=0
while :; do
  pod_state; ps=$?
  if [ "$ps" -eq 1 ]; then log "pod $POD_ID is gone; nothing left to watch"; exit 0; fi
  raw=$(read_status)
  age=$(echo "$raw" | sed -n 's/^LOGAGE \([0-9]*\)$/\1/p' | tail -1)
  st=$(echo "$raw" | sed -n 's/^STATUS: //p')
  if echo "$st" | grep -q "TERMINAL"; then
    terminate "run ended: $(echo "$st" | head -1)" && exit 0
  elif [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
    terminate "deadline backstop reached (STATUS: ${st:-none})" && exit 0
  elif [ -n "$age" ] && [ "$age" -ge $(( STALL_MIN * 60 )) ]; then
    terminate "run silent for $(( age / 60 )) min (> ${STALL_MIN}); the pod's own supervisor should have acted at 30" && exit 0
  else
    log "ok: ${st:-running}; last output ${age:-?} s ago"
  fi
  n=$(( n + 1 ))
  [ "$MAX_LOOPS" -gt 0 ] && [ "$n" -ge "$MAX_LOOPS" ] && exit 0
  sleep "$INTERVAL"
done
