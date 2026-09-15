#!/usr/bin/env bash
#
# Session B, rung 1 -- BOTH arms, data fails fast, model last.
#
#   0 setup      openpi @ 0cc8e355 + both patches, pyav, msgpack-numpy
#   1 data       chunk-010 -> slice -> merge --drop-unlabelled (compacted)
#   2 batch      one batch per arm through create_b1k_data_loader  (seconds)
#   3 norm stats per arm, then ASSERT shared statistics are identical
#   4 10 steps   per arm, PREALLOCATE=false so nvidia-smi peak VRAM is real
#
# Stops at the first failure and prints RUNG1_FAILED_AT=<stage>. Every stage
# before 4 is cheap; the ~7 GB pi05_base download happens only in stage 4.
#
# NORM STATS ARE A FULL PASS, DECODE-FREE. Decoding six camera streams per frame
# measured ~2 frames/s (hours per arm) for pixels the stats never read. Stage 3
# first PROVES decode-free == decoded on PROOF_FRAMES seeded frames (the sampler
# is seeded, so both runs see identical frames) and aborts on any difference.
#
set -uo pipefail
TASK="${TASK:-set_up_a_coffee_station_in_your_kitchen}"
CHUNK="${CHUNK:-chunk-010}"
PROOF_FRAMES="${PROOF_FRAMES:-640}"      # seeded frames used to prove decode-free == decoded
STATS_FRAMES="${STATS_FRAMES:-128000}"   # seeded sample for the real pass; see stage 3 comment
STEPS="${STEPS:-10}"
BATCHES="${BATCHES:-32 16 8}"          # tried in order; first that fits is recorded
OUT=/opt/sessionb
export PATH="${HOME}/.local/bin:${PATH}" GIT_LFS_SKIP_SMUDGE=1 UV_NO_CACHE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OPENPI_ROOT=/opt/openpi
PY=/opt/openpi/.venv/bin/python
B26=/opt/behavior26
mkdir -p "$OUT"
stage () { echo; echo "=== STAGE $1 $(date -u +%H:%M:%S)"; CUR="$1"; }
fail  () { echo "RUNG1_FAILED_AT=${CUR}: $*"; exit 1; }

stage 0_setup
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="${HOME}/.local/bin:${PATH}"
[ -d $B26/.git ] || git clone -q https://github.com/wimaan3/behavior26.git $B26 || fail clone
git -C $B26 fetch -q origin main jetson/labels && git -C $B26 checkout -q origin/main
echo "behavior26 $(git -C $B26 rev-parse --short HEAD)"
mkdir -p /opt/labels && git -C $B26 show "origin/jetson/labels:labels/${TASK}/labels.parquet" > /opt/labels/$TASK.parquet || fail labels
[ -d $OPENPI_ROOT/.git ] || git clone -q --filter=blob:none https://github.com/wensi-ai/openpi.git $OPENPI_ROOT || fail openpi_clone
git -C $OPENPI_ROOT checkout -q 0cc8e355f7bac0976db1cc3139b1ff0379feea60 || fail openpi_checkout
(cd $OPENPI_ROOT && uv sync 2>&1 | tail -2) || fail uv_sync
(cd $OPENPI_ROOT && uv pip install -q av msgpack-numpy 2>&1 | tail -2)
bash $B26/scripts/apply_openpi_patches.sh 2>&1 | tail -2
echo "patches: $(git -C $OPENPI_ROOT diff --stat | tail -1)"
$PY -c "import jax,openpi,av,msgpack_numpy;print('jax',jax.__version__,jax.devices())" || fail imports
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

stage 1_data
$PY -m huggingface_hub.commands.huggingface_cli version >/dev/null 2>&1 || true
/opt/openpi/.venv/bin/hf download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir /opt/behavior-data \
  --include "data/${CHUNK}/**" --include "meta/episodes/${CHUNK}/**" --include "videos/*/${CHUNK}/**" \
  --include "meta/info.json" --include "meta/stats.json" --include "meta/tasks.parquet" --include "meta/tasks.jsonl" \
  >/dev/null 2>&1 || fail download
echo "corpus $(du -sh /opt/behavior-data | cut -f1)"
$PY $B26/scripts/slice_task_dataset.py --source /opt/behavior-data --task "$TASK" --out /opt/tasks --overwrite 2>&1 | tail -3 || fail slice
rm -rf /opt/merged
$PY $B26/scripts/merge_progress_labels.py --dataset-root /opt/tasks/$TASK --labels /opt/labels/$TASK.parquet \
  --out-root /opt/merged --join-on global_episode_index frame_index \
  --labels-join-on episode_index frame_index --drop-unlabelled 2>&1 | grep -E "==>|dropped|compacted|FAIL|Error" | tail -8
MERGED=/opt/merged; [ -d $MERGED/data ] || MERGED=/opt/merged/$TASK
[ -f $MERGED/meta/progress_filter.json ] || fail "no progress_filter.json in merged root"
echo "MERGED=$MERGED  $(python3 -c "import json;m=json.load(open('$MERGED/meta/progress_filter.json'));i=json.load(open('$MERGED/meta/info.json'));print('manifest',m.get('episodes_out'),'info',i['total_episodes'])")"

stage 2_batch
for arm in A B; do
  if [ $arm = A ]; then CFG=pi05_b1k_frozen_vlm; X=""; else CFG=pi05_b1k_frozen_vlm_progress; X="--progress-key progress"; fi
  (cd $B26 && $PY scripts/first_batch.py --config $CFG --dataset-root $MERGED --repo-id $TASK \
     --batch-size 2 --skip-norm-stats --video-backend pyav --num-workers 0 $X 2>&1 | tail -16) | tee $OUT/batch_$arm.log
  grep -q FIRST_BATCH_OK $OUT/batch_$arm.log || fail "arm $arm batch"
done

stage 3_norm_stats
# 3a. Prove decode-free stats equal decoded stats on the SAME seeded frames.
#     Decoding all six camera streams measured ~2 frames/s; stats never read pixels.
NS="$PY $B26/scripts/compute_norm_stats_b1k.py --dataset-root $MERGED --repo-id $TASK"
rm -rf /opt/assets
if [ "${SKIP_DECODE_PROOF:-0}" = 1 ] && grep -q NORM_STATS_MATCH $OUT/ns_equivalence.log 2>/dev/null; then
  echo "DECODE_FREE_EQUIVALENT (reusing proof already logged on this box: $OUT/ns_equivalence.log)"
else
rm -rf /opt/ns_ref /opt/ns_nodecode
(cd $B26 && $NS --config-name pi05_b1k_frozen_vlm --assets-base-dir /opt/ns_ref \
   --max-frames $PROOF_FRAMES --num-workers 16 2>&1 | grep -E "config=|frames=|NORM_STATS_OK|Error" ) | tee $OUT/ns_ref.log
grep -q NORM_STATS_OK $OUT/ns_ref.log || fail "decoded reference stats"
(cd $B26 && NORM_STATS_NO_DECODE=1 $NS --config-name pi05_b1k_frozen_vlm --assets-base-dir /opt/ns_nodecode \
   --max-frames $PROOF_FRAMES --num-workers 16 2>&1 | grep -E "config=|frames=|NORM_STATS_OK|Error" ) | tee $OUT/ns_nodecode.log
grep -q NORM_STATS_OK $OUT/ns_nodecode.log || fail "decode-free stats"
$PY $B26/scripts/compare_norm_stats.py \
  --arm-a /opt/ns_ref/pi05_b1k_frozen_vlm/$TASK/norm_stats.json \
  --arm-b /opt/ns_nodecode/pi05_b1k_frozen_vlm/$TASK/norm_stats.json | tee $OUT/ns_equivalence.log
grep -q NORM_STATS_MATCH $OUT/ns_equivalence.log || fail "decode-free stats differ from decoded stats -- cannot skip decoding"
echo "DECODE_FREE_EQUIVALENT on $PROOF_FRAMES seeded frames"
fi

# 3b. The real pass: a SEEDED SAMPLE, decode-free, both arms.
#     Not all 1,247,890 frames: decode-free measured ~140 frames/s, bound by the
#     loader's single-threaded parent (pinned at ~106% CPU; more workers do not
#     help), so a full pass is ~2.5 h per arm. 128k seeded frames drawn across all
#     episodes is ample for mean/std/q01/q99, and the seed makes both arms read
#     the identical frames, so the parity assertion below stays exact.
for arm in A B; do
  if [ $arm = A ]; then CFG=pi05_b1k_frozen_vlm; else CFG=pi05_b1k_frozen_vlm_progress; fi
  T0=$(date +%s)
  (cd $B26 && NORM_STATS_NO_DECODE=1 $NS --config-name $CFG --assets-base-dir /opt/assets \
     --max-frames $STATS_FRAMES --num-workers 8 2>&1 | grep -E "config=|frames=|NORM_STATS_OK|Error" ) | tee $OUT/norm_$arm.log
  grep -q NORM_STATS_OK $OUT/norm_$arm.log || fail "arm $arm norm stats"
  echo "arm $arm full-pass wall $(( $(date +%s) - T0 ))s"
done
$PY $B26/scripts/compare_norm_stats.py \
  --arm-a /opt/assets/pi05_b1k_frozen_vlm/$TASK/norm_stats.json \
  --arm-b /opt/assets/pi05_b1k_frozen_vlm_progress/$TASK/norm_stats.json | tee $OUT/norm_compare.log
grep -q NORM_STATS_MATCH $OUT/norm_compare.log || fail "shared norm statistics differ between arms"

stage 4_ten_steps
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # MEASUREMENT ONLY: preallocation hides true peak VRAM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
for arm in A B; do
  if [ $arm = A ]; then CFG=pi05_b1k_frozen_vlm; X=""; else CFG=pi05_b1k_frozen_vlm_progress; X="--progress-key progress"; fi
  fitted=""
  for bs in $BATCHES; do
    echo "--- arm $arm  config $CFG  batch $bs  $(date -u +%H:%M:%S)"
    rm -f $OUT/vram_$arm.csv
    ( while :; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $OUT/vram_$arm.csv; sleep 2; done ) &
    SAMPLER=$!
    T0=$(date +%s)
    (cd $B26 && $PY scripts/train_b1k_rooted.py --config $CFG --exp-name rung1-$arm-bs$bs \
       --dataset-root $MERGED --repo-id $TASK --assets-base-dir /opt/assets \
       --checkpoint-base-dir /opt/checkpoints --num-train-steps $STEPS --batch-size $bs \
       --log-interval 1 --num-workers 4 --overwrite $X 2>&1) > $OUT/train_${arm}_bs$bs.log
    RC=$?
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    PEAK=$(cut -d, -f2 $OUT/vram_$arm.csv | sort -n | tail -1)
    echo "rc=$RC wall=$(( $(date +%s) - T0 ))s peak_vram_mib=$PEAK"
    grep -E "^config|^dataset_root|^norm_stats|^weight_loader|^progress_|missing|Restoring|Step [0-9]+:|TRAIN_DONE|RESOURCE_EXHAUSTED|Out of memory|Traceback|Error" \
      $OUT/train_${arm}_bs$bs.log | tail -30
    if [ $RC -eq 0 ] && grep -q TRAIN_DONE $OUT/train_${arm}_bs$bs.log; then fitted=$bs; break; fi
    grep -qE "RESOURCE_EXHAUSTED|Out of memory|OOM" $OUT/train_${arm}_bs$bs.log || fail "arm $arm batch $bs crashed for a reason other than memory"
    echo "arm $arm batch $bs OOM -- trying smaller"
  done
  [ -n "$fitted" ] || fail "arm $arm did not fit at any batch in: $BATCHES"
  echo "ARM_${arm}_FITS_AT_BATCH=$fitted PEAK_VRAM_MIB=$PEAK"
done
echo; echo "RUNG1_OK"
du -sh /opt/checkpoints /root/.cache/openpi ~/.cache/openpi 2>/dev/null; df -h / | tail -1
