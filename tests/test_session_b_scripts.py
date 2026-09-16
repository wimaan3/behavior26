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


def test_shared_paths_are_defined_outside_any_skippable_stage():
    """START_AT skips stages; a variable defined inside a skipped one is unbound
    under `set -u`. A resumed run died on 'C0: unbound variable'."""
    text = R25.read_text()
    c0 = text.index("\nC0=/opt/merged/")
    assert c0 < text.index("stage 0b_openpi_tests"), "C0 must be defined before the skippable stages"
    assert text.index("\nR1=$B26/docs/") < text.index("stage 0b_openpi_tests")


def test_resume_skips_stages_below_start_at_inclusive():
    """'>' instead of '>=' re-ran the stats stage on START_AT=4."""
    assert '"${START_AT}" -ge "$1"' in R25.read_text()


def test_a_missing_verdict_is_not_reported_as_a_difference():
    text = R25.read_text()
    assert "produced NO verdict" in text


LB = REPO / "scripts" / "loader_bench.py"


def test_loader_bench_measures_the_data_path_without_the_model():
    """The point is attribution: if it built a model, a slow result could be the
    model's fault."""
    text = LB.read_text()
    assert "create_b1k_data_loader" in text
    for forbidden in ("init_train_state", "train_step", "CheckpointWeightLoader", "create_trained_policy"):
        assert forbidden not in text, f"loader_bench must not use {forbidden}"


def test_loader_bench_discards_warmup_batches():
    """Superseded by the queue-drain test below: discarding one warm-up batch was
    never enough, because the prefetch queue outlives it."""
    text = LB.read_text()
    assert "discard" in text and "queue_depth" in text


def test_loader_bench_states_the_rate_training_needs():
    """A raw items/s number is not actionable without the bar it must clear."""
    assert "step-seconds" in LB.read_text() and "KEEPS UP" in LB.read_text()


def _fake_queue_iter(rate_items_s, batch_size, depth, clock):
    """A loader that PRODUCES at `rate_items_s` into a queue of `depth` batches,
    prefilled. Consuming a queued batch is instant; once the queue is empty the
    consumer waits for production. This is the shape that fooled the first bench.
    """
    queued = depth
    produced_at = clock()

    def nxt():
        nonlocal queued, produced_at
        if queued > 0:
            queued -= 1
            return
        wait = batch_size / rate_items_s
        produced_at = clock(advance=wait)

    return nxt


def test_measure_recovers_the_production_rate_not_the_drain_rate():
    """The defect this guards: with a prefilled queue, timing a few batches measures
    memory, not the loader. The first bench reported 777 items/s and 'KEEPS UP' for a
    loader that actually sustains ~5."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("loader_bench", LB)
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)

    now = [0.0]

    def clock(advance=0.0):
        now[0] += advance
        return now[0]

    workers, batch = 24, 32
    depth = workers * lb.PREFETCH_FACTOR
    true_rate = 5.0
    nxt = _fake_queue_iter(true_rate, batch, depth, clock)
    r = lb.measure(nxt, batches=depth + 60, batch_size=batch, queue_depth=depth,
                   clock=lambda: now[0])
    assert abs(r["items_per_s"] - true_rate) < 0.2, r


def test_measure_refuses_to_report_when_the_run_is_too_short_to_outrun_the_queue():
    """70 batches at 24 workers is 2 batches past a 72-deep queue -- not a rate."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("loader_bench", LB)
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)

    now = [0.0]

    def clock(advance=0.0):
        now[0] += advance
        return now[0]

    depth = 24 * lb.PREFETCH_FACTOR
    nxt = _fake_queue_iter(5.0, 32, depth, clock)
    r = lb.measure(nxt, batches=70, batch_size=32, queue_depth=depth, clock=lambda: now[0])
    assert r["items_per_s"] is None
    assert "too short" in r["why"], r


def test_prefetch_factor_matches_what_the_rung4_log_showed():
    """Rung 4 stalled every 24 steps at 8 workers, so the depth is workers*3, not *2.
    Using *2 leaves a third of the queue inside the 'sustained' window."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("loader_bench", LB)
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    assert lb.PREFETCH_FACTOR == 3
