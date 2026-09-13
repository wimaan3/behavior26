#!/usr/bin/env bash
#
# Two questions, one pod, four runs.
#
# Q1 REUSE -- does one evaluator invocation reload the scene per instance?
#   harness/launch.py::build_jobs says in its docstring "Scene load is per-trial
#   regardless, so grouping costs little." That is an ASSUMPTION sitting at the
#   centre of our cost model and it has never been tested. At ~9 warm minutes a
#   load it is the difference between paying once per task and once per instance
#   -- an order of magnitude on the whole A/B design.
#   Evidence it may be wrong (Discord, 2026-09): Hang-Yin describes env.reset()
#   and Evaluator.reset for resetting BETWEEN rollouts without restarting the
#   simulator, and challenge instances are *-tro_state.json STATE files rather
#   than scene files -- both point at reuse.
#
# Q2 DECOMPOSITION -- of a 555 s warm run at 51 steps, how much is load and how
#   much is stepping? 555 s is equally consistent with "540 s load, stepping ~free"
#   and "500 s load, 1.1 s/step", and at a full ~3,224-frame episode those differ
#   by an order of magnitude. Two-point subtraction: run the same thing at two
#   step counts and the fixed load cancels.
#       marginal_s_per_step = (wall_HIGH - wall_LOW) / (STEPS_HIGH - STEPS_LOW)
#   Chosen over an instrumented env wrapper because a wrapper adds per-step Python
#   work to the path whose per-step cost is the dependent variable.
#
# Every run times `python -m omnigibson.eval.eval` directly and identically, so
# the four numbers are comparable. Output goes to CONTAINER DISK: the volume has
# ~8 GB free and this writes video.
#
set -u
TASK="${TASK:-turning_on_radio}"
MODE="${MODE:-train}"
PORT="${PORT:-8000}"
SCRATCH="${ROLLOUT_SCRATCH:-/tmp}/reuse"
RESULTS="${RESULTS:-/tmp/scene_reuse.jsonl}"
STEPS_LOW="${STEPS_LOW:-50}"
STEPS_HIGH="${STEPS_HIGH:-550}"

curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null \
  || { echo "no policy server on ${PORT}"; exit 1; }

run () {  # $1 label, $2 max_steps, $3.. indices
  local label="$1" steps="$2"; shift 2
  local out="${SCRATCH}/${label}"
  rm -rf "${out}"; mkdir -p "${out}"
  echo "########## ${label}: indices=[$*] max_steps=${steps} START $(date -u +%H:%M:%S)"
  local t0 t1
  t0=$(date +%s)
  python -m omnigibson.eval.eval \
    --task-name "${TASK}" --host 127.0.0.1 --port "${PORT}" --mode "${MODE}" \
    --instance-indices "$@" \
    --num-rollouts 1 \
    --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
    --output-dir "${out}" --write-video --headless \
    --max-steps "${steps}" > "${out}/eval.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  local wall=$(( t1 - t0 ))
  local n_json; n_json=$(find "${out}/json" -name '*.json' 2>/dev/null | wc -l)
  local tot_steps; tot_steps=$(python3 -c "
import json,glob,sys
print(sum(json.load(open(f)).get('steps',0) for f in glob.glob('${out}/json/*.json')) or 0)
" 2>/dev/null || echo 0)
  echo "########## ${label} EXIT=${rc} wall_s=${wall} rollouts=${n_json} steps_total=${tot_steps} $(date -u +%H:%M:%S)"
  python3 - "$label" "$wall" "$rc" "$n_json" "$tot_steps" "$steps" "$*" >> "${RESULTS}" <<'PY'
import json, sys
label, wall, rc, n_json, tot_steps, max_steps, indices = sys.argv[1:8]
print(json.dumps({"label": label, "wall_s": int(wall), "rc": int(rc),
                  "rollouts": int(n_json), "steps_total": int(tot_steps),
                  "max_steps": int(max_steps), "indices": indices}, sort_keys=True))
PY
}

: > "${RESULTS}"
# R0 is the COLD load on a fresh pod -- kept as a datapoint, excluded from the
# two-point subtraction, which needs both points at equal cache state.
run cold_1inst_low   "${STEPS_LOW}"  0
run warm_1inst_low   "${STEPS_LOW}"  0
run warm_1inst_high  "${STEPS_HIGH}" 0
run warm_3inst_low   "${STEPS_LOW}"  0 1 2

echo "=== RESULTS ==="
cat "${RESULTS}"
python3 - "${RESULTS}" <<'PY'
import json, sys
rows = {json.loads(l)["label"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
lo, hi, three = rows.get("warm_1inst_low"), rows.get("warm_1inst_high"), rows.get("warm_3inst_low")
print()
if lo and hi and hi["steps_total"] != lo["steps_total"]:
    marg = (hi["wall_s"] - lo["wall_s"]) / (hi["steps_total"] - lo["steps_total"])
    load = lo["wall_s"] - marg * lo["steps_total"]
    print(f"Q2  marginal = {marg:.3f} s/step   implied fixed load = {load:.0f} s")
    print(f"    a 3224-frame episode would then cost {load + marg*3224:.0f} s "
          f"({(load + marg*3224)/60:.1f} min)")
if lo and three:
    print(f"\nQ1  1 instance = {lo['wall_s']} s   3 instances = {three['wall_s']} s")
    print(f"    no reuse would predict ~{3*lo['wall_s']} s; full reuse ~{lo['wall_s']} s + 2 resets")
    print(f"    ratio = {three['wall_s']/lo['wall_s']:.2f}x")
PY
echo "MEASURE_DONE"
