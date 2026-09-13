#!/usr/bin/env bash
# THE NUMBER. The single most important output of session A.
#
# Runs one rollout and reports wall clock, STEPPING FPS, and Q. Everything
# downstream -- how many experiments we can afford, whether a full submission is
# feasible, how much cloud to buy -- is planned off these figures.
#
# Reports stepping fps separately from wall clock on purpose. Scene load is a
# fixed 150-300s per trial and does not scale with episode length, so a change
# that only affects stepping is diluted in wall clock and can be missed.
#
# TRAIN MODE BY DEFAULT. The evaluator's --mode defaults to public_test, and
# this script used to omit --mode entirely -- so it evaluated instance 301, a
# SCORED public-test instance, bypassing the guard harness/launch.py enforces.
# Every public instance counts toward the reported score, so repeatedly timing
# runs there is iterating on the leaderboard. Timing is equally valid on a
# training instance.
#
# Usage:
#   bash scripts/first_rollout.sh
#   MODE=public_test INSTANCE=0 bash scripts/first_rollout.sh     # deliberate
#   THREAD_CONDITION=c bash scripts/first_rollout.sh              # see below
#
# Thread conditions (see scripts/threadfix/sitecustomize.py):
#   a  stock          -- whatever torch picks from detected cores
#   b  OMP_NUM_THREADS=4 only    -- intra-op 4, inter-op untouched
#   c  intra 4 + inter 1         -- the disabled upstream fix, in full
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-turning_on_radio}"
PORT="${PORT:-8000}"
MODE="${MODE:-train}"
INSTANCE="${INSTANCE:-0}"
COND="${THREAD_CONDITION:-a}"
# MAX_STEPS truncates the episode. For SMOKE-TESTING the plumbing only -- a
# truncated rollout's Q and fps are not comparable to a full one, so never put a
# MAX_STEPS run in the results the protocol quotes.
MAX_STEPS="${MAX_STEPS:-}"
OUT="${OUT:-rollouts/000-baseline-smoke-${COND}}"
RESULTS="${RESULTS:-${REPO}/rollouts/thread_conditions.jsonl}"

case "${COND}" in
  a) unset OMP_NUM_THREADS BEHAVIOR_TORCH_THREADS BEHAVIOR_TORCH_INTEROP || true ;;
  b) export OMP_NUM_THREADS=4
     unset BEHAVIOR_TORCH_THREADS BEHAVIOR_TORCH_INTEROP || true ;;
  c) export PYTHONPATH="${REPO}/scripts/threadfix${PYTHONPATH:+:${PYTHONPATH}}"
     export BEHAVIOR_TORCH_THREADS=4 BEHAVIOR_TORCH_INTEROP=1 ;;
  *) echo "unknown THREAD_CONDITION=${COND} (want a, b or c)" >&2; exit 2 ;;
esac

echo "==> task ${TASK}  mode ${MODE}  instance ${INSTANCE}  thread-condition ${COND}"

# Fail before the scene loads, not after: a train id that does not ship shows up
# as FileNotFoundError inside Evaluator.load_task_instance, minutes in.
python - "${TASK}" "${MODE}" "${INSTANCE}" <<'PYEOF' || exit 1
import sys, torch
task, mode, inst = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"    torch intra-op {torch.get_num_threads()}  inter-op {torch.get_num_interop_threads()}")
try:
    from omnigibson.utils.asset_utils import get_task_instance_path
    from omnigibson.eval.utils.eval_utils import get_task_cfg
    scene = get_task_cfg(task)["scene_model"]
    name = f"{scene}_task_{task}_0_{inst}_template"
    p = get_task_instance_path(scene, f"{scene}_task_{task}_instances/{name}-tro_state", mode=mode)
    if p is None:
        sys.exit(f"    ERROR: no {mode} instance {inst} for {task} -- it does not ship")
    print(f"    instance file ok")
except SystemExit:
    raise
except Exception as exc:
    print(f"    (instance pre-check skipped: {exc})")
PYEOF

echo "==> policy server must already be running on port ${PORT}"
curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null \
  || { echo "no server on ${PORT}. Start the null server or the vendor serve_b1k.py first."; exit 1; }
echo "    healthz ok"

mkdir -p "${OUT}" "$(dirname "${RESULTS}")"
START=$(date +%s)

python -m omnigibson.eval.eval \
  --task-name "${TASK}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --mode "${MODE}" \
  --instance-indices "${INSTANCE}" \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --output-dir "${OUT}" \
  --write-video \
  --headless \
  ${MAX_STEPS:+--max-steps ${MAX_STEPS}}
RC=$?

ELAPSED=$(( $(date +%s) - START ))

# Everything below reads the rollout JSON the evaluator wrote -- no re-derivation.
PYTHONPATH="${REPO}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
python - "${OUT}" "${COND}" "${ELAPSED}" "${TASK}" "${MODE}" "${INSTANCE}" "${RC}" "${RESULTS}" <<'PYEOF'
import glob, json, os, sys
out, cond, elapsed, task, mode, inst, rc, results = sys.argv[1:9]
rows = []
for p in glob.glob(os.path.join(out, "**", "*.json"), recursive=True):
    if os.path.basename(p) == "timing_manifest.json":
        continue
    try:
        rows.append(json.load(open(p)))
    except Exception:
        pass
import torch
# The environment fingerprint travels with EVERY number. Our rendering stack is
# not the organizers' (AB_PROTOCOL §3.5b) and rendering reaches Q directly --
# the rendered image is the policy's input. A Q figure without this block cannot
# be told apart from a replication failure.
try:
    from env_fingerprint import fingerprint as _fp   # via PYTHONPATH=scripts/
    _env = _fp()
except Exception as _e:
    _env = {"fingerprint_error": str(_e)}
rec = {
    "env": _env,
    "condition": cond, "task": task, "mode": mode, "instance": int(inst),
    "returncode": int(rc), "wall_s": int(elapsed),
    "max_steps": os.environ.get("MAX_STEPS") or None,   # set => smoke run, not comparable
    "intra_threads": torch.get_num_threads(),
    "inter_threads": torch.get_num_interop_threads(),
}
if rows:
    d = rows[0]
    steps = (d.get("time") or {}).get("simulator_steps")
    sim_t = (d.get("time") or {}).get("simulator_time")
    dist = d.get("agent_distance") or {}
    rec.update({
        "q_score": (d.get("q_score") or {}).get("final"),
        "success": d.get("success"),
        "sim_steps": steps, "sim_time_s": sim_t,
        "stepping_fps": round(steps / sim_t, 2) if steps and sim_t else None,
        "scene_load_s": round(int(elapsed) - sim_t, 1) if sim_t else None,
        # CONTINUOUS trajectory fields. Q on a D=1 task is binary and hides any
        # divergence short of flipping the outcome; these move every step, so two
        # runs that diverge at all will differ here even when Q agrees. That
        # matters because OMP_NUM_THREADS is a general OpenMP variable -- other
        # libraries in the Isaac Sim stack may read it, so a physics-side
        # difference can survive even if the torch policy path is bit-identical.
        # A physics-side divergence is the one that would break replication.
        "steps": d.get("steps"),
        "dist_base": dist.get("base"),
        "dist_left": dist.get("left"),
        "dist_right": dist.get("right"),
        "normalized_agent_distance": d.get("normalized_agent_distance"),
        "normalized_time": (d.get("time") or {}).get("normalized_time"),
    })
print()
print("=" * 62)
_e = rec.get("env", {})
print(f"  {'env':<16} {_e.get('os')} | {_e.get('gpu')} | drv {_e.get('driver')}")
print(f"  {'renderer':<16} {(_e.get('vulkan_devices') or ['?'])[0]} | icd rc={_e.get('vulkan_icd_negotiate_rc')}")
print(f"  {'commit':<16} {_e.get('behavior26_commit')}{' (DIRTY)' if _e.get('behavior26_dirty') else ''}")
for k in ("condition", "intra_threads", "inter_threads", "wall_s", "sim_steps",
          "sim_time_s", "stepping_fps", "scene_load_s", "q_score", "success",
          "steps", "dist_base", "dist_left", "dist_right",
          "normalized_agent_distance", "normalized_time"):
    if k in rec:
        print(f"  {k:<16} {rec[k]}")
print("=" * 62)
with open(results, "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
print(f"  appended -> {results}")
PYEOF
exit "${RC}"
