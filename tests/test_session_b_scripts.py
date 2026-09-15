"""Session B pod scripts: ordering and safety properties, pinned.

These scripts spend money when they run, and a mistake in stage order shows up
only on a rented GPU. The properties below are cheap to check here.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R25 = REPO / "scripts" / "session_b" / "rung2_5.sh"


def test_syntax():
    assert subprocess.run(["bash", "-n", str(R25)]).returncode == 0


def test_cheap_checks_run_before_any_training():
    text = R25.read_text()
    first_train = text.index("\ntrain rung4_armA")
    for marker in ("stage 0b_openpi_tests", "stage 1_data", "stage 2_batch", "stage 3_stats", "validate_roots("):
        assert text.index(marker) < first_train, f"{marker} must precede training"


def test_reused_norm_stats_are_checked_against_rung1_not_assumed():
    text = R25.read_text()
    assert "MANIFEST_SAME_AS_RUNG1" in text
    assert "9a3dcf3f7d5643913e581722133a7b878495ac19202d54e5d45ccfcd93e99d74" in text


def test_the_manifest_comparison_uses_the_same_fields_on_both_sides():
    """The first draft stripped bookkeeping keys from today's manifest only, so it
    could never match rung 1's and would have aborted every run."""
    text = R25.read_text()
    assert "for k in sys.argv[3].split()" in text


def test_two_task_stats_do_not_overwrite_single_task_stats():
    """openpi writes multi-task stats under the FIRST repo_id -- the same path as
    the coffee station's single-task stats."""
    text = R25.read_text()
    m = re.search(r"--repo-id \$\{TASKS\[0\]\} \$\{TASKS\[1\]\} --assets-base-dir (\S+)", text)
    assert m and m.group(1) != "/opt/assets"


def test_throughput_comes_from_the_arm_without_extra_backward_passes():
    text = R25.read_text()
    arm_a = [l for l in text.splitlines() if l.startswith("train rung4_armA")][0]
    arm_b = [l for l in text.splitlines() if l.startswith("train rung23_armB")][0]
    assert "--log-term-grads" not in arm_a and "--log-term-grads" in arm_b


def test_training_lines_are_timestamped():
    assert "date +%s.%N" in R25.read_text()


def test_every_training_run_is_time_boxed_and_a_protocol_run():
    text = R25.read_text()
    assert "timeout $RUN_TIMEOUT" in text and "--protocol" in text
