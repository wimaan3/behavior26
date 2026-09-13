#!/usr/bin/env bash
# Separate the three costs that a naive "first import is slow" conflates.
#
#   warm import      -- every rollout after the first on a pod pays this. It is
#                       what our workload actually costs.
#   cold NFS read    -- reading the env's files off the network volume when they
#                       are not in page cache. THIS is the only part that moving
#                       conda to container disk would fix.
#   shader compile   -- Isaac Sim's one-time compile. Cached under
#                       OMNIGIBSON_APPDATA_PATH, which is container disk by
#                       design, so it is paid ONCE PER POD and does not travel
#                       with the volume. Moving conda would not change it.
#
# Reporting only "first import took N minutes" hides which of the three it was,
# and two of them have completely different remedies.
set -uo pipefail
source /workspace/env.sh 2>/dev/null || true
export OMNIGIBSON_GPU_ID=0
REPS="${REPS:-3}"

t() {  # t <label> <python-snippet>
  local label="$1" snippet="$2"
  TIMEFORMAT="${label}=%R"
  time python -c "${snippet}" >/dev/null 2>&1
}

echo "interpreter : $(command -v python)"
echo "env         : ${CONDA_DEFAULT_ENV:-<none>}"
echo

echo "== A. warm import omnigibson (x${REPS}) -- the per-process cost we pay =="
for i in $(seq 1 "${REPS}"); do t "WARM_OMNIGIBSON_${i}" "import omnigibson"; done
echo

# Modules big enough to touch a lot of the env's file tree, chosen to avoid the
# Isaac Sim shader path entirely. First import in a fresh process reads them off
# the volume; the immediate repeat is served from page cache. The delta is the
# cold-NFS penalty, with no shader compile mixed in.
echo "== B. cold vs warm NFS read, no shader path =="
for m in scipy matplotlib sympy pandas; do
  python -c "import ${m}" >/dev/null 2>&1 || { echo "  (${m} absent, skipping)"; continue; }
  t "COLD_${m}" "import ${m}"      # may already be warm; see note below
  t "WARM_${m}" "import ${m}"
done
echo
echo "NOTE: page cache cannot be dropped inside an unprivileged container, so"
echo "'COLD_' above is only cold for modules nothing has read yet on this pod."
echo "Treat a COLD/WARM pair that is nearly equal as 'already cached', not as"
echo "'NFS is fast'. The honest cold number is the FIRST read after pod start."
echo
echo "== C. file-tree walk cost on the volume vs container disk =="
TIMEFORMAT="NFS_WALK_ENV=%R"
time find /workspace/envs/behavior -type f -name "*.py" 2>/dev/null | wc -l
TIMEFORMAT="LOCAL_WALK_ROOT=%R"
time find /usr/lib/python3 -type f -name "*.py" 2>/dev/null | wc -l
echo
echo "IMPORT_TIMING_DONE"
