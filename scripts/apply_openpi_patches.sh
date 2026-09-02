#!/usr/bin/env bash
# Apply behavior26's changes to a sibling openpi checkout.
#
# The fork is not vendored into this repo (it is ~1GB and gitignored), so our
# model changes live here as patches and are applied to the checkout on the box
# that will run them. Everything this touches is upstream code, so it verifies
# before it writes:
#
#   * the checkout is at the commit the patches were cut against, or you are
#     told exactly which commit that was;
#   * the tree is clean, so nothing of yours is silently overwritten;
#   * `git apply --check` passes before a single byte is changed;
#   * an already-patched checkout is detected and left alone, so re-running is
#     safe.
#
# Usage:
#   scripts/apply_openpi_patches.sh                  # apply
#   scripts/apply_openpi_patches.sh --check          # report, change nothing
#   scripts/apply_openpi_patches.sh --reverse        # undo
#   OPENPI_ROOT=/path/to/openpi scripts/apply_openpi_patches.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/training/patches"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/openpi}"

# The upstream commit these patches were generated against.
# wensi-ai/openpi @ branch `behavior`.
EXPECTED_BASE="0cc8e355f7bac0976db1cc3139b1ff0379feea60"

MODE="apply"
case "${1:-}" in
  --check)   MODE="check" ;;
  --reverse) MODE="reverse" ;;
  "")        ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

die()  { printf '\033[31mXX  %s\033[0m\n' "$*" >&2; exit 1; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$*" >&2; }
log()  { printf '\033[1m==> %s\033[0m\n' "$*"; }

[ -d "${OPENPI_ROOT}/.git" ] || die "OPENPI_ROOT=${OPENPI_ROOT} is not a git checkout.
    git clone -b behavior https://github.com/wensi-ai/openpi.git ${OPENPI_ROOT}"

mapfile -t PATCHES < <(find "$PATCH_DIR" -name '*.patch' | sort)
[ "${#PATCHES[@]}" -gt 0 ] || die "no patches found in ${PATCH_DIR}"

cd "$OPENPI_ROOT"
HEAD_SHA="$(git rev-parse HEAD)"

if [ "$HEAD_SHA" != "$EXPECTED_BASE" ]; then
  warn "openpi is at ${HEAD_SHA}"
  warn "patches were cut against ${EXPECTED_BASE}"
  warn "they may still apply; git apply --check below decides. If it fails, rebase"
  warn "the patches by hand -- do NOT force them, the freeze filters depend on"
  warn "exact module paths."
fi

# Already applied? Reverse-check every patch; if all reverse cleanly, the work is
# already in the tree and re-running must be a no-op rather than a conflict.
ALL_APPLIED=1
for p in "${PATCHES[@]}"; do
  git apply --reverse --check "$p" 2>/dev/null || { ALL_APPLIED=0; break; }
done

if [ "$MODE" = "reverse" ]; then
  [ "$ALL_APPLIED" = 1 ] || die "patches do not reverse cleanly; nothing undone"
  for p in "${PATCHES[@]}"; do log "reversing $(basename "$p")"; git apply --reverse "$p"; done
  log "reversed. openpi is back at upstream ${HEAD_SHA}"
  exit 0
fi

if [ "$ALL_APPLIED" = 1 ]; then
  log "already applied -- nothing to do"
  [ "$MODE" = "check" ] && exit 0
  exit 0
fi

# Refuse to patch over unrelated local edits: `git apply` would happily interleave
# them and the result would be neither version.
if ! git diff --quiet; then
  git diff --stat >&2
  die "openpi has uncommitted changes (above). Commit or stash them first --
    applying on top would mix them with ours and neither could be reversed."
fi

for p in "${PATCHES[@]}"; do
  git apply --check "$p" || die "$(basename "$p") does not apply to this checkout.
    The fork has moved. Re-derive the patch against ${HEAD_SHA} rather than
    forcing it: freeze_vlm_filter depends on exact parameter paths, and a
    half-applied patch trains the wrong parameters without erroring."
done

if [ "$MODE" = "check" ]; then
  log "all ${#PATCHES[@]} patch(es) apply cleanly. Nothing written (--check)."
  exit 0
fi

for p in "${PATCHES[@]}"; do log "applying $(basename "$p")"; git apply "$p"; done

log "applied ${#PATCHES[@]} patch(es) to ${OPENPI_ROOT}"
git -C "$OPENPI_ROOT" diff --stat
cat <<'NOTES'

==> verify before spending GPU time
    OPENPI_ROOT=<root> pytest tests/test_progress_head.py -q
    OPENPI_ROOT=<root> python scripts/estimate_memory.py

==> undo
    scripts/apply_openpi_patches.sh --reverse
NOTES
