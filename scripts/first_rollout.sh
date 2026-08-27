#!/usr/bin/env bash
# THE NUMBER. The single most important output of Week 1.
#
# Runs one rollout of the only task with a released baseline checkpoint and reports
# wall-clock. Everything downstream -- how many experiments we can afford, whether a
# full submission is feasible, how much cloud to buy -- is planned off this figure.
#
# Estimate to beat: ~20-25 min/rollout, implying ~350-420 GPU-hours for 1,000 rollouts.
set -euo pipefail

TASK="${TASK:-turning_on_radio}"
PORT="${PORT:-8000}"
OUT="${OUT:-rollouts/000-baseline-smoke}"

echo "==> policy server must already be running on port ${PORT}"
curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null \
  || { echo "no server on ${PORT}. Start the vendor serve_b1k.py first."; exit 1; }
echo "    healthz ok"

mkdir -p "${OUT}"
START=$(date +%s)

python -m omnigibson.eval.eval \
  --task-name "${TASK}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --instance-indices 0 \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir "${OUT}" \
  --write-video \
  --headless

ELAPSED=$(( $(date +%s) - START ))

echo
echo "======================================================"
printf "  wall clock        : %d s  (%.1f min)\n" "${ELAPSED}" "$(echo "${ELAPSED}/60" | bc -l)"
printf "  -> 1,000 rollouts : %.1f GPU-hours\n" "$(echo "${ELAPSED}*1000/3600" | bc -l)"
echo "======================================================"
echo
echo "  Record this in Notion -> Research Log, and in the Week 1 timing table."
echo "  Break it down: how much was scene load vs. stepping? Check ${OUT}/logs."
