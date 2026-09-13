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

echo "==> preflight  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo

# -- 1. does our own suite pass on this box at all --------------------------------------
# A green suite here means the clone is intact, python works, and the tool deps
# resolve. A red one means stop now, not after the install.
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
