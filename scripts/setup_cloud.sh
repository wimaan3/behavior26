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
set -euo pipefail

TAG="${BEHAVIOR_TAG:-v3.9.2}"
ROOT="${BEHAVIOR_ROOT:-$HOME/BEHAVIOR-1K}"

echo "==> pinned tag: ${TAG}"
echo "    confirm this is current before trusting the run"

nvidia-smi || { echo "no NVIDIA driver visible -- wrong instance type?"; exit 1; }

if [ ! -d "${ROOT}" ]; then
  git clone -b "${TAG}" https://github.com/StanfordVL/BEHAVIOR-1K.git "${ROOT}"
else
  echo "==> ${ROOT} exists; fetching ${TAG}"
  git -C "${ROOT}" fetch --tags && git -C "${ROOT}" checkout "${TAG}"
fi

cd "${ROOT}"

# --primitives pulls CuRobo, which frequently fails to build. Omit it on the first pass;
# if you need motion primitives later, install cuda-compiler=12.4 / cuda-toolkit=12.4,
# export CUDA_HOME / CUDACXX / CPATH / LIBRARY_PATH / LD_LIBRARY_PATH, then re-run with it.
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

cat <<'NOTES'

==> next
    conda activate behavior

    First OmniGibson import takes up to ~5 minutes. That is a one-time shader
    compile, not a hang.

    If it stalls at "HydraEngine rtx failed creating scene renderer":
        export OMNIGIBSON_GPU_ID=0

    PyPI packages and the Docker install are unavailable during their monorepo
    migration -- source install only.

    Then: bash scripts/first_rollout.sh
NOTES
