#!/usr/bin/env bash
#
# Why is training data-bound, and can it be fixed cheaply?
#
# Rung 4 measured: median step 3.85 s but a ~60 s stall every few steps, median GPU
# utilisation 0%, mean 6.7 s/step at 24 workers. At that rate a 30k-step arm is
# ~56 h ~ $40, so both arms of shot one cost ~$80 of a $200 budget. Halving the
# stall is worth more than any other optimisation available to us.
#
# This measures the DATA PATH ALONE -- no model, no optimiser -- so the stall is
# attributable. Three ladders, each reported in items/s and batches/s:
#
#   A. raw dataset __getitem__, single process      -> the per-item ceiling
#   B. openpi's loader at several worker counts     -> does more parallelism help?
#   C. pyav vs torchcodec decode backends           -> is the decoder the limit?
#
# A training step needs BATCH/3.85 items/s to keep the GPU busy (8.3 items/s at
# batch 32). Any configuration below that is the bottleneck, and the gap says how
# much is recoverable.
#
set -uo pipefail
TASK="${TASK:-set_up_a_coffee_station_in_your_kitchen}"
ROOT="${ROOT:-/opt/merged/$TASK}"
BATCH="${BATCH:-32}"
BATCHES="${BATCHES:-12}"
WORKER_LADDER="${WORKER_LADDER:-0 8 24 31}"
OUT="${OUT:-/opt/sessionb2/loader}"
B26=/opt/behavior26
PY=/opt/openpi/.venv/bin/python
export PATH="${HOME}/.local/bin:${PATH}" OPENPI_ROOT=/opt/openpi
# torchcodec needs FFmpeg 7 shared libraries; the PyAV wheel bundles them under
# auditwheel-hashed names, so plain sonames are symlinked and av.libs added to the
# path (its libs find their own deps through $ORIGIN).
SP=/opt/openpi/.venv/lib/python3.11/site-packages
mkdir -p /opt/ffmpeg7 "$OUT"
for f in $SP/av.libs/lib{avcodec,avformat,avutil,swresample,swscale,avdevice,avfilter}-*.so.*; do
  [ -e "$f" ] || continue; b=$(basename "$f"); stem=${b%%-*}; ver=${b#*.so.}
  ln -sf "$f" "/opt/ffmpeg7/$stem.so.${ver%%.*}"
done
export LD_LIBRARY_PATH="/opt/ffmpeg7:$SP/av.libs:${LD_LIBRARY_PATH:-}"

cd $B26
$PY scripts/loader_bench.py --dataset-root "$ROOT" --repo-id "$TASK" \
    --batch-size "$BATCH" --batches "$BATCHES" --workers $WORKER_LADDER 2>&1 | tee "$OUT/loader_bench.log"
echo "LOADER_TUNING_DONE $(date -u +%H:%M:%S)"
