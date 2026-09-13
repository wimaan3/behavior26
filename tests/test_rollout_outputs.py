"""Where the rollout writes, which is a capacity question before it is a tidiness one.

Measured 2026-09-13 on the session B box: the 150 GB network volume is at 142 GB
(og-data 91, envs 44, miniforge 4.4, BEHAVIOR-1K 2.6). **~8 GB free.** The
evaluator runs with `--write-video` at full-res RGBD, and the matrix is three
thread conditions across two tasks. Videos into 8 GB is how you fill a volume
halfway through a measurement and lose the run.

The small results file is a different artifact with different requirements: it is
what the protocol quotes, it is version-controlled, and it must stay in the repo.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROLLOUT = REPO / "scripts" / "first_rollout.sh"


def _assign(var: str) -> str:
    """The script's default for `var`, as written."""
    m = re.search(rf'^{var}="\$\{{{var}:-(.*?)\}}"', ROLLOUT.read_text(), re.M)
    assert m, f"no default assignment found for {var}"
    return m.group(1)


def test_video_output_does_not_default_onto_the_network_volume():
    """`rollouts/...` is relative to CWD, so running from /workspace/behavior26 --
    the documented location -- put full-res video on a volume with 8 GB free."""
    out = _assign("OUT")
    assert not out.startswith("rollouts/"), (
        "OUT defaults to a CWD-relative path; from the repo checkout on the volume "
        "that writes video into ~8 GB of headroom"
    )
    assert "REPO" not in out, "OUT must not be anchored to the repo -- the repo is on the volume"


def test_results_file_stays_in_the_repo():
    """The counterpart. thread_conditions.jsonl is kilobytes, is what the protocol
    quotes, and is committed -- it does NOT follow the video onto scratch."""
    results = _assign("RESULTS")
    assert "${REPO}" in results, "the results file must stay in the version-controlled repo"
    assert results.endswith(".jsonl")


def test_the_split_is_explained():
    """Without the reason this reads as inconsistency and gets 'tidied' back to
    one directory -- which is exactly the change that fills the volume."""
    text = ROLLOUT.read_text()
    assert "8 GB" in text or "8GB" in text, "cite the measurement that forced the split"
    assert "--write-video" in text
