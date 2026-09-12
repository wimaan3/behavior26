#!/usr/bin/env bash
# Provision a fresh cloud GPU box for BEHAVIOR-1K. Idempotent -- safe to re-run.
#
# Requirements (from the official install page):
#   Ubuntu 22.04+ or Windows 10+, 32GB+ RAM, NVIDIA RTX 2070+, 8GB+ VRAM.
#   Your laptops do not meet this. Start on cloud; do not wait for the 3090.
#
# The pinned tag MOVES. It was v3.9.1, then v3.9.2. Check the evaluation page
# before every fresh clone and again before submitting:
#   https://behavior.stanford.edu/challenge/evaluation.html
#
# ============================================================================
# EVERYTHING EXPENSIVE LIVES ON THE NETWORK VOLUME. THIS IS NOT A STYLE CHOICE.
# ============================================================================
# The conda env (~10s of GB) and the BEHAVIOR-1K dataset (behavior-1k-assets is
# 29.3 GB zipped) cost real time and money to build. Container disk dies with the
# pod. Put them on the volume and a second pod mounting the same volume starts
# with the stack already there.
#
# That is what makes the card comparison cheap: bring up an A5000 ($0.27/hr)
# beside the 4090 ($0.74/hr), mount the same volume, re-run rollouts, and let
# measured seconds-per-rollout decide which card runs ALL evaluation. Evaluation
# is the dominant cost line, so that decision is worth more than the pods. It is
# only minutes of work if the environment is already on the volume.
#
# *** A CONDA ENV IS NOT PATH-RELOCATABLE. ***
# Absolute paths are baked into the env at creation time -- script shebangs,
# conda-meta/*.json, pkgs records, compiled extensions' RPATHs. Moving or
# re-mounting it elsewhere does not "mostly work"; it breaks in ways that look
# like unrelated import errors. So the env must be CREATED at its final path,
# and EVERY pod must mount the volume at the SAME mount point. Do not
# "simplify" this back to `conda create -n behavior` on container disk, and do
# not create it in one place and move it.
#
# Upstream's setup.sh takes `--new-env NAME` and runs `conda create -n "$NAME"`
# (setup.sh l.252) -- it has no prefix option. Rather than patch a script that
# moves under us, we point CONDA_ENVS_PATH at the volume: conda creates named
# envs in the first writable entry, so `-n behavior` lands on the volume and
# `conda activate behavior` keeps working. No upstream patch to maintain.
#
# What deliberately does NOT go on the volume: OMNIGIBSON_APPDATA_PATH.
# OmniGibson's own macros.py says of it -- "on HPC clusters like SLURM you
# ideally want to put this in a local path rather than a networked filesystem
# where read/writes are slow and race conditions can occur". A RunPod network
# volume is exactly that. It is a cache; losing it with the pod is fine.
set -euo pipefail

TAG="${BEHAVIOR_TAG:-v3.9.2}"

# /workspace, CONFIRMED 2026-09-12 -- and set EXPLICITLY on every pod rather than
# relying on RunPod's default, because this path is now load-bearing: the env on
# the volume was created here and cannot be moved (see path-relocatable above).
# A pod that mounts the volume anywhere else does not get a degraded env, it
# gets a broken one. Set it in the pod config; do not assume the default holds.
VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"

ROOT="${BEHAVIOR_ROOT:-${VOLUME_ROOT}/BEHAVIOR-1K}"
ENVS_DIR="${VOLUME_ROOT}/envs"
ENV_NAME="${BEHAVIOR_ENV_NAME:-behavior}"
ENV_PREFIX="${ENVS_DIR}/${ENV_NAME}"
OG_DATA="${OMNIGIBSON_DATA_PATH:-${VOLUME_ROOT}/og-data}"
OG_APPDATA="${OMNIGIBSON_APPDATA_PATH:-${HOME}/og-appdata}"   # container disk, on purpose
ENV_SH="${VOLUME_ROOT}/env.sh"

echo "==> pinned tag: ${TAG}"
echo "    confirm this is current before trusting the run"

# -- the guard that stops the whole point of this being silently lost ----------
if [ ! -d "${VOLUME_ROOT}" ]; then
  echo "ERROR: VOLUME_ROOT=${VOLUME_ROOT} does not exist." >&2
  echo "       Attach the network volume, or pass the real mount point:" >&2
  echo "         VOLUME_ROOT=/your/mount bash scripts/setup_cloud.sh" >&2
  exit 1
fi

# A network volume is a separate filesystem. If VOLUME_ROOT sits on the same
# device as /, it is container disk wearing the right name -- and everything we
# build would die with the pod, which we would only discover on the next pod.
if [ "$(stat -c %d / 2>/dev/null)" = "$(stat -c %d "${VOLUME_ROOT}" 2>/dev/null)" ]; then
  echo "ERROR: ${VOLUME_ROOT} is on the same filesystem as / -- that is CONTAINER DISK," >&2
  echo "       not a network volume. The conda env and the 29.3 GB dataset would be" >&2
  echo "       destroyed with this pod, and the A5000 comparison would be impossible." >&2
  echo "       Check the volume is attached and mounted where you think it is." >&2
  echo "       Override only if you really mean it: ALLOW_CONTAINER_DISK=1" >&2
  [ "${ALLOW_CONTAINER_DISK:-0}" = "1" ] || exit 1
  echo "!! ALLOW_CONTAINER_DISK=1 -- proceeding on container disk anyway." >&2
fi

if ! touch "${VOLUME_ROOT}/.write-test" 2>/dev/null; then
  echo "ERROR: ${VOLUME_ROOT} is not writable." >&2
  exit 1
fi
rm -f "${VOLUME_ROOT}/.write-test"

echo "==> volume layout"
echo "    env      ${ENV_PREFIX}"
echo "    repo     ${ROOT}"
echo "    dataset  ${OG_DATA}"
echo "    appdata  ${OG_APPDATA}   (container disk -- cache, deliberately not on the volume)"

nvidia-smi || { echo "no NVIDIA driver visible -- wrong instance type?"; exit 1; }

mkdir -p "${ENVS_DIR}" "${OG_DATA}" "${OG_APPDATA}"

# conda resolves `-n NAME` against CONDA_ENVS_PATH, so this is what puts the env
# on the volume without patching upstream.
export CONDA_ENVS_PATH="${ENVS_DIR}"
export OMNIGIBSON_DATA_PATH="${OG_DATA}"
export OMNIGIBSON_APPDATA_PATH="${OG_APPDATA}"

# Written to the VOLUME so any later shell -- including the second pod's, which
# must not re-run this script -- gets the same environment by sourcing one file.
cat > "${ENV_SH}" <<EOF
# Source this in every shell on any pod mounting this volume.
#   source ${ENV_SH}
# The conda env here was created at this exact path and cannot be moved.
export CONDA_ENVS_PATH="${ENVS_DIR}"
export OMNIGIBSON_DATA_PATH="${OG_DATA}"
export OMNIGIBSON_APPDATA_PATH="${OG_APPDATA}"
export BEHAVIOR_ROOT="${ROOT}"
conda activate ${ENV_NAME} 2>/dev/null || echo "run: conda activate ${ENV_NAME}"
EOF
echo "==> wrote ${ENV_SH}"

# Convenience for THIS pod only; ~/.bashrc is container disk and does not persist.
if ! grep -qsF "source ${ENV_SH}" "${HOME}/.bashrc" 2>/dev/null; then
  echo "source ${ENV_SH}" >> "${HOME}/.bashrc"
fi

if [ ! -d "${ROOT}" ]; then
  git clone -b "${TAG}" https://github.com/StanfordVL/BEHAVIOR-1K.git "${ROOT}"
else
  echo "==> ${ROOT} exists; fetching ${TAG}"
  git -C "${ROOT}" fetch --tags && git -C "${ROOT}" checkout "${TAG}"
fi

cd "${ROOT}"

# Second pod, or a re-run: the env is already on the volume. Upstream's setup.sh
# hard-errors on an existing env name (setup.sh l.242-244), so do not call it.
# This is the path the A5000 comparison takes -- it should cost seconds.
if [ -d "${ENV_PREFIX}" ]; then
  echo "==> ${ENV_PREFIX} already exists -- reusing the env on the volume."
  echo "    Skipping install. If you genuinely want a rebuild, remove that"
  echo "    directory first and re-run."
else
  # --primitives pulls CuRobo, which frequently fails to build. Omit it on the first pass;
  # if you need motion primitives later, install cuda-compiler=12.4 / cuda-toolkit=12.4,
  # export CUDA_HOME / CUDACXX / CPATH / LIBRARY_PATH / LD_LIBRARY_PATH, then re-run with it.
  ./setup.sh --new-env "${ENV_NAME}" --omnigibson --bddl --joylo --dataset --eval \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
fi

cat <<NOTES

==> next
    source ${ENV_SH}          # sets CONDA_ENVS_PATH etc, then activates

    First OmniGibson import takes up to ~5 minutes. That is a one-time shader
    compile, not a hang.

    If it stalls at "HydraEngine rtx failed creating scene renderer":
        export OMNIGIBSON_GPU_ID=0

    PyPI packages and the Docker install are unavailable during their monorepo
    migration -- source install only.

    Then: bash scripts/first_rollout.sh

==> on a SECOND pod against this same volume (the A5000 comparison)
    Mount the volume at ${VOLUME_ROOT} EXPLICITLY -- the same path, set in the
    pod config rather than left to the default. A different mount point does not
    degrade the env, it breaks it. Then just:
        source ${ENV_SH}
    Do not re-run this script's install; it will detect the env and skip.
    Compare seconds-per-rollout over SEVERAL instances per card, not one:
    the evaluator is nondeterministic and a single rollout's wall-clock varies.

    This pod pays the ~5 min shader compile again on its first OmniGibson
    import: OMNIGIBSON_APPDATA_PATH is container disk on purpose, so the shader
    cache does not travel with the volume. Expected, not a fault -- and time the
    SECOND import, not the first (see docs/AB_PROTOCOL.md, session A).
NOTES
