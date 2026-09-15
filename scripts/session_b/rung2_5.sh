#!/usr/bin/env bash
#
# Session B, rungs 2-5 in ONE pod session.
#
#   0 setup     openpi @ 0cc8e355 + patches (incl. term-gradient logging)
#   1 data      BOTH dev-loop tasks: slice -> merge --drop-unlabelled -> compacted
#   2 batch     regression: one batch per arm, plus a TWO-task arm-B batch
#   3 stats     coffee: reuse rung 1's committed stats IFF this box's merge
#               manifest is identical to rung 1's; two-task stats for rung 5
#   4 RUNGS 2-4 arm A and arm B, TRAIN_STEPS each (default 1000 = end of LR warmup)
#               A: clean -> throughput (rung 4) and the control's action_loss
#               B: --log-term-grads -> progress learning (rung 2) + lambda (rung 3)
#   5 RUNG 5    MULTI_STEPS on the two-task mix, same config as arm A, to test
#               whether steps/sec depends on task count
#
# Training runs do not abort the session on failure: each records its rc and the
# next runs, so one bad rung cannot cost the others' results. Prerequisite stages
# (0-3) do abort.
#
# Every training output line is prefixed with a unix timestamp, because
# train_b1k.py prints "Step N: ..." with no time and steps/sec would otherwise be
# a guess from total wall clock. GPU utilisation is sampled alongside: low GPU
# utilisation at steady state means the step is DATA-bound (rung 1 found the
# loader's parent process pinned at ~106% CPU).
#
set -uo pipefail
TASKS=(set_up_a_coffee_station_in_your_kitchen putting_shoes_on_rack)
CHUNKS=(chunk-010 chunk-022)
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
MULTI_STEPS="${MULTI_STEPS:-200}"
RUN_TIMEOUT="${RUN_TIMEOUT:-4800}"          # per training run, seconds
MULTI_STATS_FRAMES="${MULTI_STATS_FRAMES:-32000}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"                     # TrainConfig's own default, so throughput is representative
OUT=/opt/sessionb2
export PATH="${HOME}/.local/bin:${PATH}" GIT_LFS_SKIP_SMUDGE=1 UV_NO_CACHE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OPENPI_ROOT=/opt/openpi
PY=/opt/openpi/.venv/bin/python
B26=/opt/behavior26
mkdir -p "$OUT"
stage () { echo; echo "=== STAGE $1 $(date -u +%H:%M:%S)"; CUR="$1"; }
fail  () { echo "RUNG25_FAILED_AT=${CUR}: $*"; exit 1; }
TSTAMP='while IFS= read -r l; do printf "%s %s\n" "$(date +%s.%N)" "$l"; done'

stage 0_setup
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="${HOME}/.local/bin:${PATH}"
[ -d $B26/.git ] || git clone -q https://github.com/wimaan3/behavior26.git $B26 || fail clone
git -C $B26 fetch -q origin main && git -C $B26 checkout -q origin/main
echo "behavior26 $(git -C $B26 rev-parse --short HEAD)"
[ -d $OPENPI_ROOT/.git ] || git clone -q --filter=blob:none https://github.com/wensi-ai/openpi.git $OPENPI_ROOT || fail openpi_clone
git -C $OPENPI_ROOT checkout -q 0cc8e355f7bac0976db1cc3139b1ff0379feea60 || fail openpi_checkout
(cd $OPENPI_ROOT && uv sync 2>&1 | tail -1) || fail uv_sync
(cd $OPENPI_ROOT && uv pip install -q av msgpack-numpy 2>&1 | tail -1)
bash $B26/scripts/apply_openpi_patches.sh 2>&1 | tail -1
grep -q log_loss_term_grad_norms $OPENPI_ROOT/src/openpi/training/config.py || fail "patch 0002 lacks term-gradient logging"
$PY -c "import jax,openpi,av,msgpack_numpy;print('jax',jax.__version__,jax.devices())" || fail imports
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee $OUT/gpu.txt

stage 0b_openpi_tests
# The JAX tests cannot run on the laptop. Run them here before spending training
# time -- includes the per-term gradient decomposition the lambda calibration reads.
(cd $B26 && OPENPI_ROOT=$OPENPI_ROOT $PY -m pytest tests/test_progress_head.py -q -p no:cacheprovider 2>&1 | tail -3) | tee $OUT/openpi_tests.log
grep -qE "[0-9]+ passed" $OUT/openpi_tests.log && ! grep -qE "[0-9]+ failed|error" $OUT/openpi_tests.log || fail "progress-head tests failed on real JAX"

stage 1_data
for i in 0 1; do
  T=${TASKS[$i]}; C=${CHUNKS[$i]}
  /opt/openpi/.venv/bin/hf download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir /opt/behavior-data \
    --include "data/${C}/**" --include "meta/episodes/${C}/**" --include "videos/*/${C}/**" \
    --include "meta/info.json" --include "meta/stats.json" --include "meta/tasks.parquet" --include "meta/tasks.jsonl" \
    >/dev/null 2>&1 || fail "download $C"
  $PY $B26/scripts/slice_task_dataset.py --source /opt/behavior-data --task "$T" --out /opt/tasks --overwrite 2>&1 | tail -1 || fail "slice $T"
  rm -rf /opt/merged/$T
  $PY $B26/scripts/merge_progress_labels.py --dataset-root /opt/tasks/$T --labels $B26/labels/$T/labels.parquet \
    --out-root /opt/merged/$T --join-on global_episode_index frame_index \
    --labels-join-on episode_index frame_index --drop-unlabelled 2>&1 | grep -E "dropped|compacted|FAIL" | tail -3
done
(cd $B26/scripts && $PY -c "
from b1k_roots import validate_roots
p = validate_roots('/opt/merged', ['${TASKS[0]}', '${TASKS[1]}'], progress_key='progress', protocol=True)
print('ROOTS_OK', [d.name for d in p.dataset_dirs])
") | tee $OUT/roots.log
grep -q ROOTS_OK $OUT/roots.log || fail "merged roots failed validation"
du -sh /opt/behavior-data /opt/merged | tee -a $OUT/roots.log

stage 2_batch
C0=/opt/merged/${TASKS[0]}
for spec in "A|pi05_b1k_frozen_vlm|$C0|${TASKS[0]}|" \
            "B|pi05_b1k_frozen_vlm_progress|$C0|${TASKS[0]}|--progress-key progress" \
            "B2|pi05_b1k_frozen_vlm_progress|/opt/merged|${TASKS[0]} ${TASKS[1]}|--progress-key progress"; do
  IFS='|' read -r arm cfg root ids extra <<< "$spec"
  (cd $B26 && $PY scripts/first_batch.py --config $cfg --dataset-root $root --repo-id $ids \
     --batch-size 2 --skip-norm-stats --video-backend pyav --num-workers 0 $extra 2>&1 | tail -14) | tee $OUT/batch_$arm.log
  grep -q FIRST_BATCH_OK $OUT/batch_$arm.log || fail "batch $arm"
done

stage 3_stats
R1=$B26/docs/sessionB-2026-09-15-rung1
# Same data as rung 1? Compare the fields that describe WHAT was kept -- identical
# on both sides -- rather than whole files, whose bookkeeping keys have grown.
SAME_KEYS='dropped_episodes rows_in rows_out unlabelled_rows label_column join_on labels_join_on episodes_out'
python3 - "$R1/progress_filter.json" "$C0/meta/progress_filter.json" "$SAME_KEYS" <<'PY2' | tee $OUT/manifest_vs_rung1.log
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
diff = {k: (a.get(k), b.get(k)) for k in sys.argv[3].split() if a.get(k) != b.get(k)}
print("MANIFEST_SAME_AS_RUNG1" if not diff else f"MANIFEST_DIFFERS {diff}")
PY2
grep -q MANIFEST_SAME_AS_RUNG1 $OUT/manifest_vs_rung1.log \
  || fail "coffee merge differs from rung 1 -- committed norm stats may not describe this data"
echo "coffee merge identical to rung 1 on kept-data fields -> reusing committed norm stats"
for cfg in pi05_b1k_frozen_vlm pi05_b1k_frozen_vlm_progress; do
  mkdir -p /opt/assets/$cfg/${TASKS[0]}
  cp $R1/norm_stats/$cfg.norm_stats.json /opt/assets/$cfg/${TASKS[0]}/norm_stats.json
done
sha256sum /opt/assets/*/${TASKS[0]}/norm_stats.json | tee $OUT/norm_sha.txt
[ "$(cut -d' ' -f1 $OUT/norm_sha.txt | sort -u | wc -l)" = 1 ] || fail "arms' reused norm stats differ"
grep -q 9a3dcf3f7d5643913e581722133a7b878495ac19202d54e5d45ccfcd93e99d74 $OUT/norm_sha.txt || fail "reused stats sha is not rung 1's"
# Two-task stats for rung 5 in a SEPARATE assets dir: openpi writes multi-task stats
# under the FIRST repo_id, which is the same path as coffee's single-task stats.
(cd $B26 && NORM_STATS_NO_DECODE=1 $PY scripts/compute_norm_stats_b1k.py --config-name pi05_b1k_frozen_vlm \
   --dataset-root /opt/merged --repo-id ${TASKS[0]} ${TASKS[1]} --assets-base-dir /opt/assets_2task \
   --max-frames $MULTI_STATS_FRAMES --num-workers 8 2>&1 | grep -E "config=|frames=|NORM_STATS_OK|FAIL|Error") | tee $OUT/norm_2task.log
grep -q NORM_STATS_OK $OUT/norm_2task.log || fail "two-task norm stats"

train () {   # name config root "ids" assets steps extra...
  local name=$1 cfg=$2 root=$3 ids=$4 assets=$5 steps=$6; shift 6
  echo "--- RUN $name ($cfg, tasks: $ids, $steps steps) $(date -u +%H:%M:%S)"
  ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')" >> $OUT/gpu_$name.csv; sleep 5; done ) &
  local sampler=$!
  local t0=$(date +%s)
  (cd $B26 && timeout $RUN_TIMEOUT $PY -u scripts/train_b1k_rooted.py --config $cfg --exp-name $name \
     --dataset-root $root --repo-id $ids --assets-base-dir $assets --checkpoint-base-dir /opt/checkpoints \
     --num-train-steps $steps --batch-size $BATCH --log-interval 1 --num-workers $WORKERS \
     --protocol --overwrite "$@" 2>&1 | eval "$TSTAMP") > $OUT/train_$name.log
  local rc=${PIPESTATUS[0]}
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  echo "RUN_${name}_RC=$rc wall=$(( $(date +%s) - t0 ))s steps_logged=$(grep -c 'Step [0-9]*:' $OUT/train_$name.log) TRAIN_DONE=$(grep -c TRAIN_DONE $OUT/train_$name.log)"
  rm -rf /opt/checkpoints/*/$name     # disposable; keeps the 200 GB disk from filling
}

stage 4_rungs_2_to_4
train rung4_armA pi05_b1k_frozen_vlm          $C0 "${TASKS[0]}" /opt/assets $TRAIN_STEPS
train rung23_armB pi05_b1k_frozen_vlm_progress $C0 "${TASKS[0]}" /opt/assets $TRAIN_STEPS --progress-key progress --log-term-grads

stage 5_rung5
train rung5_2task pi05_b1k_frozen_vlm /opt/merged "${TASKS[0]} ${TASKS[1]}" /opt/assets_2task $MULTI_STEPS

echo; echo "RUNG25_DONE $(date -u +%H:%M:%S)"
df -h / | tail -1
