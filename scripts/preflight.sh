#!/usr/bin/env bash
# Cheap checks that catch expensive failures. Run BEFORE setup_cloud.sh.
#
# setup_cloud.sh is 30-60 minutes and pulls ~30 GB. Everything below takes
# seconds and fails for reasons that would otherwise surface after that spend --
# or worse, after the first rollout dies on step 1 with a KeyError.
#
# Exits non-zero on any failure. Safe to re-run.
#
# NOTE: there is deliberately no tests/test_preflight.py. This script RUNS
# `pytest tests/`, so a pytest test that invoked it would recurse forever. Its
# failure paths were verified by hand (banned tag, unresolvable tag, wrong robot
# name) -- if you change the checks, re-verify them the same way rather than
# "adding the missing test".
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIRED_TAG="${BEHAVIOR_TAG:-v3.9.2}"
BANNED_TAGS="v3.9.0"          # not permitted for evaluation
FAILED=0

pass() { printf "  \033[32mok\033[0m    %s\n" "$1"; }
fail() { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAILED=1; }
warn() { printf "  \033[33mwarn\033[0m  %s\n" "$1"; }

# Tests that INVOKE preflight must not run inside preflight's own pytest call --
# that recurses forever. They skip on this marker.
export PREFLIGHT_RUNNING=1

echo "==> preflight  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo

# -- 0. what this box ACTUALLY has ------------------------------------------------------
# nproc and free report the HOST. A container is bounded by its cgroup, and the gap
# is not small: the session-A pod showed 192 CPUs / 723 GB against a real 20.4 /
# 84 GB. Printing the host numbers here would actively mislead whoever sizes a
# sweep, so print the limits and say what nproc claims only to contradict it.
echo "0. box resources (cgroup limits, not nproc/free)"
CPUQ=""
if [ -f /sys/fs/cgroup/cpu.max ]; then                       # cgroup v2
  read -r _q _p < /sys/fs/cgroup/cpu.max
  [ "${_q}" != "max" ] && CPUQ="$(awk -v q="${_q}" -v p="${_p}" 'BEGIN{printf "%.1f", q/p}')"
elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then        # cgroup v1
  _q="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  _p="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)"
  [ "${_q}" -gt 0 ] 2>/dev/null && CPUQ="$(awk -v q="${_q}" -v p="${_p}" 'BEGIN{printf "%.1f", q/p}')"
fi
if [ -n "${CPUQ}" ]; then
  pass "CPU quota   ${CPUQ} cores   (nproc claims $(nproc) -- do NOT size workers by nproc)"
else
  warn "CPU quota   unlimited/unknown; nproc claims $(nproc)"
  CPUQ="$(nproc)"
fi

MEML=""
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -f "$f" ] || continue
  _m="$(cat "$f")"
  case "${_m}" in max|"") ;; *) [ "${_m}" -lt 1000000000000000 ] 2>/dev/null \
      && MEML="$(awk -v m="${_m}" 'BEGIN{printf "%.0f", m/1073741824}')" ;;
  esac
done
if [ -n "${MEML}" ]; then
  pass "memory      ${MEML} GB      (free claims $(free -g | awk '/^Mem:/{print $2}') GB)"
else
  warn "memory      unlimited/unknown; free claims $(free -g | awk '/^Mem:/{print $2}') GB"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  pass "GPU         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi

# Each eval worker is a full OmniGibson instance. Upstream's TORCH_NUM_THREADS is
# None in v3.9.2, so torch sizes its pool from detected cores and ignores the
# cgroup -- pin it per worker, then divide.
THREADS="${OMP_NUM_THREADS:-4}"
warn "eval workers: ~$(awk -v c="${CPUQ}" -v t="${THREADS}" 'BEGIN{printf "%d", (c/t)}') concurrent (${CPUQ} cores / ${THREADS} threads per worker)"
warn "  --workers defaults to 1. Pin OMP_NUM_THREADS=${THREADS} per worker:"
warn "  upstream evaluator.py has TORCH_NUM_THREADS = None, so torch would"
warn "  otherwise size its pool from $(nproc) detected cores and oversubscribe."
echo

# -- 1. does our own suite pass on this box at all --------------------------------------
# A green suite here means the clone is intact, python works, and the tool deps
# resolve. A red one means stop now, not after the install.
# -- 0b. CAN THIS BOX RENDER? -----------------------------------------------------------
# The single check that would have replaced an entire evening on 2026-09-13.
# See scripts/vulkan_check.py for why nvidia-smi and torch.cuda both say nothing
# about this, and why the failure otherwise surfaces as a segfault minutes into a
# scene load -- after the install and the 36 GB download are already paid for.
echo "0b. rendering (Vulkan)"
if command -v vulkaninfo >/dev/null 2>&1 || [ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
  if python3 "${REPO}/scripts/vulkan_check.py"; then
    :
  else
    FAILED=1
  fi
else
  warn "no vulkaninfo and no libGLX_nvidia -- cannot verify rendering here"
  warn "  (fine on a laptop; on a GPU box install vulkan-tools and re-run)"
fi
echo

# -- 0c. BUILD TOOLCHAIN ----------------------------------------------------------------
# `import omnigibson` invokes torch.compile/TorchInductor, which shells out to a
# C++ compiler to build CPU kernels AT IMPORT TIME. No compiler means:
#
#   torch._inductor.exc.InductorError: InvalidCxxCompiler: No working C++ compiler
#
# which mentions nothing about the environment and surfaces ~100 minutes into a
# fresh install, at the dataset step, after the 42 GB env is already built.
# Measured 2026-09-13 on selkies-egl-desktop:26.04, which ships gcc but NOT g++ --
# a C compiler is not sufficient, and a desktop image is not a dev image.
#
# Fix needs no root: conda install -c conda-forge cxx-compiler -p <env>
echo "0c. build toolchain"
CXX_FOUND=""
for c in "${CXX:-}" g++ c++ clang++ x86_64-conda-linux-gnu-g++; do
  [ -n "${c}" ] && command -v "${c}" >/dev/null 2>&1 && { CXX_FOUND="$(command -v "${c}")"; break; }
done
if [ -n "${CXX_FOUND}" ]; then
  pass "C++ compiler: ${CXX_FOUND}"
else
  fail "NO C++ COMPILER. torch.compile will fail at 'import omnigibson'."
  echo "        gcc alone is not enough -- Inductor needs a C++ driver."
  echo "        Fix without root:"
  echo "          conda install -y -c conda-forge cxx-compiler -p \$CONDA_PREFIX"
fi
command -v gcc >/dev/null 2>&1 || warn "no C compiler (gcc) either"
echo

# -- 0d. LATE-FAILURE TRAPS -------------------------------------------------------------
# Every check here is ~1 second and guards a step costing 30-100 minutes. They
# share a shape: the failure lands AFTER the expensive part, and reports itself
# as something other than what it is.
echo "0d. late-failure traps"

# /dev/shm -- Docker defaults it to 64 MB. PyTorch DataLoader workers share
# tensors through it, so a small one kills TRAINING (session B, the expensive
# one) with "Bus error" or "No space left on device" -- neither of which mentions
# shared memory. Classic, and it does not show up until workers spin up.
SHM_B="$(df -B1 /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
if [ -n "${SHM_B}" ]; then
  SHM_GB="$(awk -v b="${SHM_B}" 'BEGIN{printf "%.1f", b/1073741824}')"
  if awk -v b="${SHM_B}" 'BEGIN{exit !(b < 1073741824)}'; then
    fail "/dev/shm is only ${SHM_GB} GB -- DataLoader workers will die with a bus error"
    echo "        Docker's default is 64 MB. Recreate the pod with a larger --shm-size,"
    echo "        or run training with num_workers=0 (slow) as a stopgap."
  elif awk -v b="${SHM_B}" 'BEGIN{exit !(b < 4294967296)}'; then
    warn "/dev/shm is ${SHM_GB} GB -- fine for eval, tight for training workers"
  else
    pass "/dev/shm ${SHM_GB} GB"
  fi
else
  warn "/dev/shm not found -- cannot check shared memory"
fi

# ulimit -n -- Isaac Sim opens a great many files (USD assets, extensions). The
# default 1024 runs out PARTWAY THROUGH SCENE LOAD, with an error about whichever
# file happened to be next rather than about the limit.
NOFILE_S="$(ulimit -Sn 2>/dev/null)"; NOFILE_H="$(ulimit -Hn 2>/dev/null)"
if [ "${NOFILE_S}" = "unlimited" ] || { [ -n "${NOFILE_S}" ] && [ "${NOFILE_S}" -ge 4096 ] 2>/dev/null; }; then
  pass "open-file limit ${NOFILE_S} (hard ${NOFILE_H})"
else
  fail "open-file limit is only ${NOFILE_S} -- Isaac Sim will fail mid scene load"
  echo "        Raise it: ulimit -n ${NOFILE_H:-65536}   (hard limit is ${NOFILE_H:-unknown})"
fi

# Free space for the 36 GB dataset, checked BEFORE the download rather than after.
# NOTE: on a RunPod network volume, df reports the whole shared cluster (2.3 PB
# observed), NOT our quota -- so a plausible-looking "439T free" means nothing.
# Detect that and say so rather than passing a check we did not actually make.
VOL="${VOLUME_ROOT:-/workspace}"
if [ -d "${VOL}" ]; then
  VOL_TOT_K="$(df -k "${VOL}" 2>/dev/null | awk 'NR==2{print $2}')"
  VOL_AVAIL_K="$(df -k "${VOL}" 2>/dev/null | awk 'NR==2{print $4}')"
  if [ -n "${VOL_TOT_K}" ] && [ "${VOL_TOT_K}" -gt 10737418240 ] 2>/dev/null; then
    warn "df on ${VOL} reports the shared cluster, not our quota -- cannot verify free space"
    warn "  the dataset needs ~36 GB; check the volume size in the RunPod console"
  elif [ -n "${VOL_AVAIL_K}" ] && [ "${VOL_AVAIL_K}" -lt 41943040 ] 2>/dev/null; then
    fail "only $(awk -v k="${VOL_AVAIL_K}" 'BEGIN{printf "%.1f", k/1048576}') GB free on ${VOL}; the dataset needs ~36 GB"
  elif [ -n "${VOL_AVAIL_K}" ]; then
    pass "${VOL} has $(awk -v k="${VOL_AVAIL_K}" 'BEGIN{printf "%.0f", k/1048576}') GB free (dataset needs ~36 GB)"
  fi
fi

# /tmp -- Isaac Sim extracts there, and some images mount a tiny tmpfs.
TMPD="${TMPDIR:-/tmp}"
if touch "${TMPD}/.preflight-$$" 2>/dev/null; then
  rm -f "${TMPD}/.preflight-$$"
  TMP_AVAIL_K="$(df -k "${TMPD}" 2>/dev/null | awk 'NR==2{print $4}')"
  if [ -n "${TMP_AVAIL_K}" ] && [ "${TMP_AVAIL_K}" -lt 2097152 ] 2>/dev/null; then
    fail "${TMPD} has only $(awk -v k="${TMP_AVAIL_K}" 'BEGIN{printf "%.1f", k/1048576}') GB -- Isaac Sim extracts there"
  else
    pass "${TMPD} writable, $(awk -v k="${TMP_AVAIL_K:-0}" 'BEGIN{printf "%.0f", k/1048576}') GB free"
  fi
else
  fail "${TMPD} is not writable"
fi

# The env's Python must be 3.11. If it ever resolves to the host's (3.14 on this
# image) almost nothing has prebuilt wheels and pip compiles from source -- which
# looks exactly like a slow install but takes HOURS rather than minutes.
PYV="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
PYWHICH="$(command -v python 2>/dev/null)"
# Only ASSERT 3.11 when we are actually in the behavior env. On a dev laptop any
# python runs the test suite fine, and a red preflight there teaches people to
# ignore it -- which would cost us the checks that matter.
IN_ENV=0
case "${PYWHICH}" in
  *"/envs/${BEHAVIOR_ENV_NAME:-behavior}/bin/python") IN_ENV=1 ;;
esac
[ "${CONDA_DEFAULT_ENV:-}" = "${BEHAVIOR_ENV_NAME:-behavior}" ] && IN_ENV=1

if [ "${PYV}" = "3.11" ]; then
  pass "env python ${PYV} (${PYWHICH})"
elif [ -n "${PYV}" ] && [ "${IN_ENV}" -eq 0 ]; then
  warn "python is ${PYV} (${PYWHICH}) -- not the behavior env"
  warn "  fine locally; on the GPU box run 'source \${VOLUME_ROOT:-/workspace}/env.sh' first"
elif [ -n "${PYV}" ]; then
  fail "behavior env python is ${PYV}, expected 3.11 -- ${PYWHICH}"
  echo "        On the wrong interpreter almost nothing has prebuilt wheels and pip"
  echo "        builds from source: hours, not minutes, looking like a slow install."
  echo "        Did you 'source ${VOL}/env.sh'?"
else
  warn "no python on PATH"
fi
echo

# -- 0f. DID THE DATASET ACTUALLY LAND? -------------------------------------------------
# Same shape as the g++ trap in 0c: an empty og-data does not fail at import, it
# fails at SCENE LOAD -- after the box is already billing -- with an error that
# names a missing USD file rather than a missing download.
#
# `-d` is not the check. setup_cloud.sh creates og-data itself, so the directory
# exists from the first run onward; the case we actually hit was the directory
# present and EMPTY, because the installer had been skipped. Check for contents.
echo "0f. dataset"
OG_DIR="${OMNIGIBSON_DATA_PATH:-${VOLUME_ROOT:-/workspace}/og-data}"
if [ ! -d "${OG_DIR}" ]; then
  warn "no ${OG_DIR} yet -- setup_cloud.sh has not run here"
elif [ -z "$(ls -A "${OG_DIR}" 2>/dev/null)" ]; then
  fail "${OG_DIR} exists but is EMPTY -- the dataset never downloaded"
  echo "        Scene load will fail on a missing USD, not on a missing dataset."
  echo "        Fix: bash scripts/download_dataset.sh   (idempotent, ~36 GB)"
else
  pass "dataset present at ${OG_DIR} ($(du -sh "${OG_DIR}" 2>/dev/null | cut -f1))"
fi
echo

# -- 0e. SESSION B: can jax actually train here? ----------------------------------------
# Asserted now rather than discovered during the first PAID training run. jax is
# openpi's backend; a CUDA/driver pair it does not support leaves it silently on
# CPU, which does not crash -- it trains ~100x too slow, the same failure shape as
# the llvmpipe trap in 0b. Skipped cleanly when openpi/jax is not installed yet,
# because session A does not need it.
echo "0e. session B readiness (jax)"
if python -c "import jax" >/dev/null 2>&1; then
  JAX_OUT="$(python - <<'JAXPROBE' 2>/dev/null
import jax
devs = jax.devices()
kinds = sorted({d.platform for d in devs})
print(f"{jax.__version__}|{','.join(kinds)}|{len(devs)}|{devs[0] if devs else 'none'}")
JAXPROBE
)"
  JAX_VER="${JAX_OUT%%|*}"; JAX_REST="${JAX_OUT#*|}"
  JAX_KINDS="${JAX_REST%%|*}"; JAX_REST="${JAX_REST#*|}"
  JAX_N="${JAX_REST%%|*}"; JAX_DEV="${JAX_REST#*|}"
  case "${JAX_KINDS}" in
    *gpu*|*cuda*|*rocm*)
      pass "jax ${JAX_VER} sees ${JAX_N} accelerator(s): ${JAX_DEV}" ;;
    *)
      fail "jax ${JAX_VER} sees only CPU (${JAX_KINDS}) -- training would be ~100x too slow"
      echo "        It will NOT crash; it will quietly train on CPU. Check the"
      echo "        jax[cuda] build against this host's CUDA/driver pair." ;;
  esac
else
  warn "jax not installed -- session B readiness unchecked (fine for session A)"
fi
echo

echo "1. our test suite"
if ! python3 -c "import pandas, yaml, pyarrow" 2>/dev/null; then
  warn "tool deps missing -- installing requirements-tools.txt"
  python3 -m pip install -q -r "${REPO}/requirements-tools.txt" 2>&1 | tail -3
fi
if (cd "${REPO}" && python3 -m pytest tests/ -q -p no:cacheprovider >/tmp/preflight-pytest.log 2>&1); then
  pass "$(tail -1 /tmp/preflight-pytest.log)"
else
  fail "pytest failed -- see /tmp/preflight-pytest.log"
  tail -15 /tmp/preflight-pytest.log | sed 's/^/        /'
fi

# A GREEN SUITE IS NOT FULL COVERAGE. A module-level pytest.importorskip does not
# report as a skip -- the module fails to COLLECT and its tests vanish from the
# run entirely, so the summary still says "all passed" with a smaller number
# nobody reads. That is how a missing pyarrow hid 20 tests on this box while the
# dev machine ran them all. Assert every test file yields at least one test.
COLLECTED="$(cd "${REPO}" && python3 -m pytest tests/ -q -p no:cacheprovider \
              --collect-only 2>/dev/null | grep '^tests/' | sed 's/::.*//' | sort -u)"
MISSING=""
for f in "${REPO}"/tests/test_*.py; do
  rel="tests/$(basename "$f")"
  printf '%s\n' "${COLLECTED}" | grep -qx "${rel}" || MISSING="${MISSING} ${rel}"
done
if [ -n "${MISSING}" ]; then
  fail "test module(s) collected ZERO tests -- a dependency is missing, not a skip:${MISSING}"
  echo "        their tests are silently absent from the run above; install the missing dep"
else
  pass "all $(printf '%s\n' "${COLLECTED}" | wc -l) test modules collected"
fi
echo

# -- 2. the BEHAVIOR-1K tag ------------------------------------------------------------
# The pinned tag MOVES (it was v3.9.1, then v3.9.2) and v3.9.0 is not permitted
# for evaluation. Getting this wrong means a full reinstall, or a submission
# scored against the wrong evaluator.
echo "2. BEHAVIOR-1K tag"
for banned in ${BANNED_TAGS}; do
  if [ "${REQUIRED_TAG}" = "${banned}" ]; then
    fail "BEHAVIOR_TAG=${REQUIRED_TAG} is BANNED for evaluation"
  fi
done
if [ "${REQUIRED_TAG}" = "v3.9.2" ]; then
  pass "tag ${REQUIRED_TAG}"
else
  warn "tag ${REQUIRED_TAG} is not v3.9.2 -- confirm against the evaluation page:"
  warn "  https://behavior.stanford.edu/challenge/evaluation.html"
fi
# Catch a retired or mistyped tag now rather than 40 minutes into a clone.
if git ls-remote --tags --exit-code https://github.com/StanfordVL/BEHAVIOR-1K.git \
     "refs/tags/${REQUIRED_TAG}" >/dev/null 2>&1; then
  pass "tag ${REQUIRED_TAG} exists upstream"
else
  fail "tag ${REQUIRED_TAG} does not resolve upstream -- setup_cloud.sh would fail on clone"
fi
# If the checkout already exists, it is the thing that actually matters.
CLONE="${BEHAVIOR_ROOT:-${VOLUME_ROOT:-/workspace}/BEHAVIOR-1K}"
if [ -d "${CLONE}/.git" ]; then
  ACTUAL="$(git -C "${CLONE}" describe --tags --exact-match 2>/dev/null || echo "<not on a tag>")"
  if [ "${ACTUAL}" = "${REQUIRED_TAG}" ]; then
    pass "checkout at ${CLONE} is on ${ACTUAL}"
  else
    fail "checkout at ${CLONE} is on ${ACTUAL}, expected ${REQUIRED_TAG}"
  fi
fi
echo

# -- 3. the robot name ------------------------------------------------------------------
# The evaluator's r1pro.yaml says `name: robot_r1`; upstream openpi's b1k.py says
# `name="robot"`. Every observation key is prefixed with it, so a mismatch means
# eval_b1k_wrapper looks up `robot::proprio` while the evaluator publishes
# `robot_r1::proprio` -- KeyError on the FIRST STEP of the FIRST ROLLOUT, after
# the scene has loaded and the money is spent.
echo "3. robot name (robot_r1)"
YAML="${REPO}/configs/robot/r1pro.yaml"
YAML_NAME="$(awk '/^name:/{print $2; exit}' "${YAML}" 2>/dev/null)"
if [ "${YAML_NAME}" = "robot_r1" ]; then
  pass "configs/robot/r1pro.yaml -> name: ${YAML_NAME}"
else
  fail "configs/robot/r1pro.yaml -> name: ${YAML_NAME:-<missing>}, expected robot_r1"
fi

PATCH="$(ls "${REPO}"/training/patches/*.patch 2>/dev/null | head -1)"
if [ -n "${PATCH}" ] && grep -q 'ROBOT_NAME = "robot_r1"' "${PATCH}"; then
  pass "openpi patch sets ROBOT_NAME = \"robot_r1\""
else
  fail "the openpi patch does not set ROBOT_NAME = \"robot_r1\" -- the fix is missing"
fi

# Once openpi is present and importable, the real comparison is a test we already
# have; it skips silently without openpi, which is exactly when it matters least.
if (cd "${REPO}" && python3 -m pytest tests/test_robot_config.py -q -p no:cacheprovider \
      >/tmp/preflight-robot.log 2>&1); then
  if grep -qE "[0-9]+ passed" /tmp/preflight-robot.log && ! grep -q "skipped" /tmp/preflight-robot.log; then
    pass "live openpi<->evaluator name check ran and passed"
  else
    warn "live openpi<->evaluator name check SKIPPED (openpi not importable here)."
    warn "  Re-run this after the openpi fork is installed -- it is the check that"
    warn "  actually proves the first step will not KeyError."
  fi
else
  fail "tests/test_robot_config.py failed -- see /tmp/preflight-robot.log"
  tail -15 /tmp/preflight-robot.log | sed 's/^/        /'
fi
echo

if [ "${FAILED}" -ne 0 ]; then
  echo "==> PREFLIGHT FAILED. Do not run setup_cloud.sh until these are fixed."
  exit 1
fi
echo "==> preflight clean. Safe to run setup_cloud.sh."
