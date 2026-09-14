#!/usr/bin/env bash
#
# Install the openpi fork and the released turning_on_radio baseline, on
# CONTAINER DISK.
#
# WHY CONTAINER DISK: the network volume is at 142/150 GB (og-data 91, envs 44).
# openpi's venv plus a pi05 checkpoint does not fit, and per the 2026-09-13
# storage decision heavy write paths belong on local disk anyway.
# Cost: this is reinstalled per pod. Accepted -- it is minutes, and the volume
# has no room.
#
# THE CHECKPOINT. The openpi fork's docs describe only fine-tuning and its
# serve example points --policy.dir at your own training output, so the fork
# looks like it has no baseline. It does -- the CHALLENGE publishes one, for
# turning_on_radio only:
#   https://behavior.stanford.edu/challenge/baselines.html
# That is the single task with a released checkpoint, which is why the whole
# session-A design is pinned to turning_on_radio.
#
set -uo pipefail
OPENPI_ROOT="${OPENPI_ROOT:-/opt/openpi}"
CKPT_ROOT="${CKPT_ROOT:-/opt/baseline}"
# wensi-ai/openpi @ branch `behavior`; the commit our patches were cut against.
OPENPI_COMMIT="${OPENPI_COMMIT:-0cc8e355f7bac0976db1cc3139b1ff0379feea60}"
GDRIVE_ID="${GDRIVE_ID:-1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2}"

say () { echo "=== $* ($(date -u +%H:%M:%S))"; }

say "STAGE 1: uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "STAGE1_EXIT=1"; exit 1; }
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null || { echo "STAGE1_EXIT=1 uv not on PATH"; exit 1; }
echo "uv: $(uv --version)"

say "STAGE 2: clone openpi @ ${OPENPI_COMMIT:0:8}"
if [ ! -d "${OPENPI_ROOT}/.git" ]; then
  git clone --filter=blob:none https://github.com/wensi-ai/openpi.git "${OPENPI_ROOT}" \
    || { echo "STAGE2_EXIT=1"; exit 1; }
fi
git -C "${OPENPI_ROOT}" fetch --all -q
git -C "${OPENPI_ROOT}" checkout -q "${OPENPI_COMMIT}" 2>/dev/null \
  || git -C "${OPENPI_ROOT}" checkout -q behavior
echo "openpi HEAD: $(git -C "${OPENPI_ROOT}" rev-parse --short HEAD)"

say "STAGE 3: uv sync (GIT_LFS_SKIP_SMUDGE required -- lerobot is an LFS dep)"
cd "${OPENPI_ROOT}"
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -5 || { echo "STAGE3_EXIT=1"; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . 2>&1 | tail -3 || { echo "STAGE3_EXIT=1"; exit 1; }
GIT_LFS_SKIP_SMUDGE=1 uv pip install -q gdown 2>&1 | tail -2

say "STAGE 4: baseline checkpoint"
mkdir -p "${CKPT_ROOT}"
if [ -z "$(ls -A "${CKPT_ROOT}" 2>/dev/null)" ]; then
  # Google Drive, so gdown rather than curl: large files need the confirm-token
  # handshake that a plain GET does not do (you get an HTML warning page instead,
  # which then fails to untar with a misleading "not in gzip format").
  # NO --fuzzy: gdown 6.2.0 does not have that flag (it is absent from --help,
  # which is what makes "unrecognized arguments: --fuzzy" confusing -- the flag
  # exists in older/newer docs). 6.2.0 takes url_or_id positionally and resolves
  # a bare file id itself. `--` still ends uv's own argument list.
  uv run -- gdown "${GDRIVE_ID}" \
       -O "${CKPT_ROOT}/baseline.download" 2>&1 | tail -4 \
    || { echo "STAGE4_EXIT=1 gdown failed"; exit 1; }
  say "unpacking $(du -sh "${CKPT_ROOT}/baseline.download" | cut -f1)"
  case "$(file -b --mime-type "${CKPT_ROOT}/baseline.download")" in
    application/zip)  unzip -q "${CKPT_ROOT}/baseline.download" -d "${CKPT_ROOT}" ;;
    application/gzip|application/x-gzip) tar xzf "${CKPT_ROOT}/baseline.download" -C "${CKPT_ROOT}" ;;
    application/x-tar) tar xf "${CKPT_ROOT}/baseline.download" -C "${CKPT_ROOT}" ;;
    *) echo "!! unexpected type $(file -b "${CKPT_ROOT}/baseline.download") -- inspect by hand" ;;
  esac
fi
echo "checkpoint tree:"
find "${CKPT_ROOT}" -maxdepth 3 -type d | head -20
echo "total: $(du -sh "${CKPT_ROOT}" | cut -f1)"
echo "INSTALL_EXIT=0"
