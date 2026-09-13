#!/usr/bin/env bash
#
# Does scene reuse survive n=27, or only n=3?
#
# 2026-09-13 measured 3 instances in one evaluator invocation at 719 s against
# 720 s for one -- the scene is loaded once and reused. That result collapsed the
# evaluation budget (k=8 becomes ~$2.40) and it is load-bearing for every
# scheduling decision downstream. But it was verified at n=3 with RSS ~13.7 GB,
# and the real sweep is n=27.
#
# If memory grows per instance, 27 either OOMs or thrashes, the saving is limited
# to small n, and the k=8 case evaporates. So: run the real thing and watch RSS.
#
# PASS requires all four:
#   1. exit 0
#   2. 27 rollout JSONs with 27 distinct instance_ids
#   3. wall_s within ~1.5x of a single-instance warm run (one load, not 27)
#   4. peak RSS bounded -- NOT linear in instances completed
#
# Criterion 4 is the subtle one: a run can pass 1-3 and still be unusable at
# n=54 if RSS climbs steadily. The sampler writes a time series so the SHAPE is
# inspectable, not just the peak.
#
set -u
TASK="${TASK:-turning_on_radio}"
MODE="${MODE:-train}"
PORT="${PORT:-8000}"
N="${N:-27}"
STEPS="${STEPS:-50}"
OUT="${ROLLOUT_SCRATCH:-/tmp}/scale"
SAMPLES="${SAMPLES:-/tmp/scale_rss.csv}"

curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null || { echo "no policy server"; exit 1; }

rm -rf "${OUT}"; mkdir -p "${OUT}"
INDICES=$(seq 0 $((N - 1)))

# Sampler: RSS of the eval process, GPU memory, and how many rollouts have landed.
# Rollout count is what lets us plot memory AGAINST progress rather than time.
cat > /tmp/rss_sampler.sh <<'SAMP'
#!/usr/bin/env bash
echo "iso,rss_gb,vmpeak_gb,gpu_mib,rollouts_done" > "$1"
while true; do
  PID=$(pgrep -f 'omnigibson.eval.eval' | head -1)
  [ -z "$PID" ] && { sleep 5; continue; }
  RSS=$(awk '/VmRSS/{printf "%.2f", $2/1048576}' /proc/$PID/status 2>/dev/null)
  PEAK=$(awk '/VmPeak/{printf "%.2f", $2/1048576}' /proc/$PID/status 2>/dev/null)
  GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  DONE=$(find "$2/json" -name '*.json' 2>/dev/null | wc -l)
  printf '%s,%s,%s,%s,%s\n' "$(date -u +%H:%M:%S)" "${RSS:-}" "${PEAK:-}" "${GPU:-}" "${DONE:-0}" >> "$1"
  sleep 10
done
SAMP
chmod +x /tmp/rss_sampler.sh
setsid nohup /tmp/rss_sampler.sh "${SAMPLES}" "${OUT}" >/dev/null 2>&1 </dev/null &
SAMPLER=$!

echo "########## n=${N} START $(date -u +%H:%M:%S)"
T0=$(date +%s)
python -m omnigibson.eval.eval \
  --task-name "${TASK}" --host 127.0.0.1 --port "${PORT}" --mode "${MODE}" \
  --instance-indices ${INDICES} \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir "${OUT}" --write-video --headless \
  --max-steps "${STEPS}" > "${OUT}/eval.log" 2>&1
RC=$?
WALL=$(( $(date +%s) - T0 ))
kill "${SAMPLER}" 2>/dev/null; pkill -f rss_sampler.sh 2>/dev/null

echo "########## n=${N} EXIT=${RC} wall_s=${WALL} $(date -u +%H:%M:%S)"
python3 - "${OUT}" "${SAMPLES}" "${WALL}" "${RC}" "${N}" <<'PY'
import glob, json, sys
out, samples, wall, rc, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
files = sorted(glob.glob(f"{out}/json/*.json"))
ids, steps = [], 0
for f in files:
    d = json.load(open(f)); ids.append(d.get("instance_id")); steps += d.get("steps") or 0
print(f"\nrollouts written : {len(files)} / {n}")
print(f"distinct ids     : {len(set(ids))}  -> {sorted(set(ids))}")
print(f"steps total      : {steps}")
print(f"wall_s           : {wall}   rc={rc}")
rows = [l.strip().split(',') for l in open(samples)][1:] if samples else []
rows = [r for r in rows if len(r) >= 5 and r[1]]
if rows:
    rss = [float(r[1]) for r in rows]
    done = [int(r[4]) for r in rows]
    print(f"RSS GB           : start {rss[0]:.1f}  peak {max(rss):.1f}  end {rss[-1]:.1f}")
    print(f"GPU MiB          : peak {max(int(r[3]) for r in rows if r[3])}")
    # Growth per completed rollout: the number that decides whether n=54 is safe.
    if done and max(done) > 1:
        first = next((r for r, d in zip(rss, done) if d >= 1), rss[0])
        last  = rss[-1]
        per   = (last - first) / max(max(done) - 1, 1)
        print(f"RSS growth       : {per:+.3f} GB per completed rollout")
        print(f"  extrapolated to n=54: {last + per*27:.1f} GB")
print("\nSCALE_DONE")
PY
