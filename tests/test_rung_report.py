"""The rung report's rules, pinned against synthetic logs. Pure Python."""
from __future__ import annotations

import math
import random

from analysis.rung_report import branch0, paired_action, parse, throughput


def _log(n, *, progress=None, action=None, dt=0.5, t0=1000.0):
    lines = []
    for s in range(n):
        parts = [f"action_loss={action(s):.4f}" if action else "action_loss=0.8"]
        if progress is not None:
            parts.append(f"progress_loss={progress(s)}")
        lines.append(f"{t0 + s * dt:.3f} Step {s}: " + ", ".join(parts))
    return "\n".join(lines)


def test_parse_reads_timestamped_step_lines():
    run = parse(_log(3, progress=lambda s: 0.69))
    assert run.steps == [0, 1, 2] and run.times[1] == 1000.5
    assert run.series("progress_loss") == [0.69, 0.69, 0.69]


def test_absent_progress_loss_is_a_broken_run():
    assert branch0(parse(_log(300)))["verdict"] == "BROKEN"


def test_nan_is_a_broken_run():
    assert branch0(parse(_log(300, progress=lambda s: "nan" if s == 150 else 0.69)))["verdict"] == "BROKEN"


def test_noisy_but_flat_is_broken():
    rng = random.Random(0)
    run = parse(_log(1000, progress=lambda s: round(0.693 + rng.gauss(0, 0.01), 4)))
    assert branch0(run)["verdict"] == "BROKEN"


def test_a_real_decrease_is_learning():
    rng = random.Random(1)
    run = parse(_log(1000, progress=lambda s: round(0.693 - 0.2 * s / 1000 + rng.gauss(0, 0.01), 4)))
    r = branch0(run)
    assert r["verdict"] == "LEARNING" and r["z"] > 2


def test_too_short_is_inconclusive_not_a_verdict():
    assert branch0(parse(_log(150, progress=lambda s: 0.69)))["verdict"] == "INCONCLUSIVE"


def test_identical_action_losses_are_not_degraded():
    a = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s)))
    b = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s), progress=lambda s: 0.69))
    assert paired_action(a, b)["verdict"] == "NOT_DEGRADED"


def test_consistently_higher_action_loss_in_B_is_degraded():
    a = parse(_log(300, action=lambda s: 0.8 + 0.1 * math.sin(s)))
    b = parse(_log(300, action=lambda s: 0.85 + 0.1 * math.sin(s), progress=lambda s: 0.69))
    assert paired_action(a, b)["verdict"] == "DEGRADED"


def test_throughput_excludes_compile_steps():
    """Step 0-49 slow (compile), then 0.5 s/step."""
    lines = []
    t = 0.0
    for s in range(300):
        t += 20.0 if s < 50 else 0.5
        lines.append(f"{t:.3f} Step {s}: action_loss=0.8")
    r = throughput(parse("\n".join(lines)))
    assert abs(r["steps_per_s"] - 2.0) < 1e-6


def test_a_second_log_line_for_the_same_step_does_not_misalign_series():
    """The lambda calibration writes its own line for the same step with different
    keys. Zipping steps against a filtered series would shift every later step."""
    from analysis.rung_report import by_step
    text = "\n".join([
        "100.0 Step 0: action_loss=1.0",
        "100.1 Step 0: grad_norm_action=2.0, grad_norm_progress_raw=4.0",
        "101.0 Step 1: action_loss=2.0",
        "102.0 Step 2: action_loss=3.0",
    ])
    run = parse(text)
    assert by_step(run, "action_loss") == {0: 1.0, 1: 2.0, 2: 3.0}
    assert by_step(run, "grad_norm_action") == {0: 2.0}


REAL_LINE = ("1789517953.480086811 \r\rStep 55: action_loss=0.7019, grad_norm=1.2206, "
             "loss=0.7019, param_norm=1801.9882")


def test_tqdm_carriage_returns_do_not_hide_every_step():
    """Real logs carry \\r before the step text. The first parser found ZERO steps in
    a 1000-step log while passing on \\r-free fixtures."""
    run = parse(REAL_LINE + "\n" + REAL_LINE.replace("Step 55", "Step 56").replace("1789517953", "1789517957"))
    assert run.steps == [55, 56]
    assert run.times == [1789517953.480086811, 1789517957.480086811]
    assert run.series("action_loss") == [0.7019, 0.7019]


def test_lambda_parser_also_survives_carriage_returns():
    from analysis.lambda_calibration import parse_log
    line = ("1789.0 \r\rStep 0: grad_norm_action=2.0, grad_norm_progress_raw=4.0, "
            "grad_cosine_action_progress=0.01")
    rows = parse_log(line)
    assert rows and rows[0]["grad_norm_action"] == 2.0


def test_reading_a_log_file_must_not_translate_carriage_returns(tmp_path):
    """read_text() turns a lone \\r into \\n at READ time, splitting the timestamp
    away from the step. A real 1000-step log parsed to zero steps that way."""
    from analysis.rung_report import read_log
    p = tmp_path / "train.log"
    p.write_bytes((REAL_LINE + "\n").encode())
    assert parse(p.read_text()).steps == [], "read_text loses it -- this is the trap"
    assert parse(read_log(p)).steps == [55], "read_log must preserve it"


def test_throughput_reports_mean_as_well_as_median():
    """Step time is bimodal (fast step + periodic loader stall). The median
    describes the fast path; the MEAN sets wall clock and cost."""
    lines, t = [], 0.0
    for s in range(300):
        t += 60.0 if s % 10 == 0 else 4.0      # one stall every ten steps
        lines.append(f"{t:.3f} Step {s}: action_loss=0.8")
    r = throughput(parse("\n".join(lines)))
    assert abs(r["s_per_step"] - 4.0) < 0.01
    assert r["mean_s_per_step"] > 9.0
    assert r["wall_steps_per_s"] < r["steps_per_s"]
