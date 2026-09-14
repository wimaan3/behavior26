#!/usr/bin/env bash
#
# Block until the policy server answers /healthz, or fail with a reason.
#
# THE BUG THIS REPLACES. The obvious loop is:
#
#     for i in $(seq 1 90); do
#       curl healthz && break
#       pgrep -f serve_b1k || { echo SERVER_DIED; exit 1; }
#       sleep 10
#     done
#
# and it reports SERVER_DIED on a perfectly healthy server. The launcher is
# `bash serve_baseline.sh`, which execs `uv run ...`, which then spawns python.
# For the first seconds NOTHING on the box has "serve_b1k" in its argv, so the
# liveness probe fires before the thing it is probing exists. Measured
# 2026-09-14: false SERVER_DIED at t=10s while the server went on to load a
# 6.2 GiB checkpoint and serve normally.
#
# Two fixes, both needed:
#   * a GRACE period before any liveness probe counts;
#   * probe the LAUNCHER pattern, not just the final process name.
#
# Same class of error as the log-watcher that read a byte count as a done-count:
# a detector that is wrong in the direction of "your working job is dead".
#
set -uo pipefail
PORT="${PORT:-8000}"
PATTERN="${PATTERN:-serve_b1k|serve_baseline}"
GRACE="${GRACE:-60}"       # seconds before a missing process counts as death
TIMEOUT="${TIMEOUT:-900}"  # a 6 GiB checkpoint restore is not instant

# Exclude ourselves and our ancestors from the liveness match.
#
# `pgrep -f serve_b1k` also matches any PARENT whose command line happens to
# contain the pattern -- the shell that launched us, a CI wrapper, an inline
# `PATTERN=serve_b1k bash wait...`. Such a probe reports "alive" forever and can
# never detect death, which is the silent half of the same bug: the loud half
# says a working server died, this half says a dead one lives.
ancestors () {
  local pid=$$
  while [ "${pid}" -gt 1 ]; do
    echo "${pid}"
    pid=$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null) || break
    [ -n "${pid}" ] || break
  done
}

server_alive () {
  local excl match
  excl=" $(ancestors | tr '\n' ' ') "
  for match in $(pgrep -f "${PATTERN}" 2>/dev/null); do
    case "${excl}" in *" ${match} "*) continue ;; esac
    return 0
  done
  return 1
}

START=$(date +%s)
while :; do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null)" = "200" ]; then
    echo "SERVER_UP after $(( $(date +%s) - START ))s"; exit 0
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "${ELAPSED}" -ge "${GRACE}" ] && ! server_alive; then
    echo "SERVER_DIED after ${ELAPSED}s (no process matching ${PATTERN})"; exit 1
  fi
  if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
    echo "SERVER_TIMEOUT after ${ELAPSED}s"; exit 1
  fi
  sleep 5
done
