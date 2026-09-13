#!/usr/bin/env bash
#
# Pull the BEHAVIOR-1K assets onto the network volume.
#
# WHY THIS IS ITS OWN SCRIPT, and not a branch inside setup_cloud.sh:
# the conda env and the 36 GB dataset are two independent artifacts that happen
# to live on the same volume, and they fail independently. setup_cloud.sh used
# to gate BOTH on `if [ -d "${ENV_PREFIX}" ]`, because upstream's setup.sh does
# the env and the dataset in one invocation and hard-errors on an existing env
# name. That cost us a night: the first install died at the dataset step (the
# image ships no g++, and `import omnigibson` runs TorchInductor). After the
# compiler fix the env existed, so the re-run "succeeded" in seconds with
# og-data still at 512 bytes -- a silent partial install whose only symptom is a
# scene that fails to load, much later, on a box that costs money by the hour.
#
# So: guard the dataset on the DATASET's presence. One condition per artifact.
#
# The three calls below are lifted verbatim from upstream setup.sh's `--dataset`
# block (v3.9.2, lines ~480-497). If upstream renames one, tests/test_setup_cloud.py
# fails and points here.
#
#   bash scripts/download_dataset.sh              # idempotent; skips if present
#   FORCE=1 bash scripts/download_dataset.sh      # re-download anyway
#
set -uo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
OG_DATA="${OMNIGIBSON_DATA_PATH:-${VOLUME_ROOT}/og-data}"
ENV_SH="${ENV_SH:-${VOLUME_ROOT}/env.sh}"

# The dataset is "present" if og-data has any entry at all. A stricter check
# (a manifest, a size floor) would be better, but the download is not atomic and
# we have no upstream manifest to check against -- so this catches the case we
# actually hit (empty) and FORCE=1 covers a torn one.
if [ "${FORCE:-0}" != "1" ] && [ -d "${OG_DATA}" ] && [ -n "$(ls -A "${OG_DATA}" 2>/dev/null)" ]; then
  echo "==> dataset already present at ${OG_DATA} -- skipping download."
  echo "    FORCE=1 to re-download."
  exit 0
fi

echo "==> og-data is empty (${OG_DATA}); downloading the dataset."

# Only source env.sh if the current shell cannot already import omnigibson --
# setup_cloud.sh calls this with the environment set up, and sourcing env.sh
# there would pay the ~45 s import check twice.
if ! python -c "import omnigibson" >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  [ -f "${ENV_SH}" ] && . "${ENV_SH}"
fi

export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH="${OG_DATA}"
mkdir -p "${OG_DATA}"

echo "    python: $(command -v python)"
echo "    OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH}"

# Each step costs a fresh ~45 s `import omnigibson` on top of its transfer. That
# is deliberate: one process per step means a failure names the step it died in,
# which the single upstream invocation did not.
step () {   # $1 label, $2 python one-liner
  echo "=== dataset step $1 START $(date -u +%H:%M:%S) ==="
  python -c "$2"
  local rc=$?
  echo "=== dataset step $1 EXIT=${rc} $(date -u +%H:%M:%S) ==="
  if [ "${rc}" -ne 0 ]; then
    echo "ERROR: dataset step $1 failed (rc=${rc})" >&2
    echo "DATASET_EXIT=${rc}"
    exit "${rc}"
  fi
}

step robot_assets \
  "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
step b1k_assets \
  "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
step task_instances_2026 \
  "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"

echo "==> dataset on volume: $(du -sh "${OG_DATA}" 2>/dev/null | cut -f1)"
echo "DATASET_EXIT=0"
