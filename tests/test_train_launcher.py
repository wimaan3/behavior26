"""The training launcher, pinned against three defects found reading openpi.

1. `DataConfigFactory.base_config` is `tyro.conf.Suppress[...]`, so
   `--data.base_config.dataset_root` is NOT a CLI flag -- tyro rejects it at
   parse. train_cloud.sh passed exactly that to train_b1k.py and to
   compute_norm_stats.py. The root has to be set programmatically.
2. `assets_base_dir` defaults to "./assets", RELATIVE TO CWD. Norm stats written
   from one directory are silently not found by training launched from another,
   and openpi logs "Norm stats not found ... skipping" rather than failing.
3. `log_interval` defaults to 100, so a 10-step run logs only step 0 -- useless
   for rung 3's per-step action_loss / progress_loss calibration.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "scripts" / "train_b1k_rooted.py"
NORM = REPO / "scripts" / "compute_norm_stats_b1k.py"
TRAIN = REPO / "scripts" / "train_cloud.sh"


def test_launcher_sets_the_root_programmatically_and_calls_the_real_trainer():
    text = LAUNCH.read_text()
    assert "dataclasses.replace" in text and "dataset_root" in text
    assert "train_b1k" in text and ".main(" in text, "must run openpi's own training loop"


def test_launcher_refuses_a_missing_or_untaken_root():
    text = LAUNCH.read_text()
    assert "does not exist" in text
    assert "resolved to" in text, "a root that silently did not take must fail too"


def test_assets_dir_is_absolute_in_both_writer_and_reader():
    """Relative ./assets is how norm stats get written and never read."""
    for f in (LAUNCH, NORM):
        text = f.read_text()
        assert "assets_base_dir" in text, f"{f.name} must pin assets_base_dir"
        assert "is_absolute" in text, f"{f.name} must refuse a relative assets dir"


def test_launcher_can_log_every_step():
    assert "log_interval" in LAUNCH.read_text()


def test_train_cloud_no_longer_passes_a_suppressed_tyro_field():
    text = TRAIN.read_text()
    assert "--data.base_config.dataset_root" not in text, (
        "base_config is tyro.conf.Suppress; that flag is rejected at parse"
    )


def test_train_cloud_does_not_let_tee_hide_a_training_failure():
    """`cmd | tee log` reports tee's status. Without pipefail a crashed training
    run looks like success and the next line checks for a checkpoint dir that a
    previous run may have left behind."""
    text = TRAIN.read_text()
    assert re.search(r"set -[a-z]*o pipefail|set -o pipefail", text), "train_cloud.sh needs pipefail"


def test_launcher_can_turn_on_term_gradient_logging_and_refuses_a_stale_patch():
    text = LAUNCH.read_text()
    assert "--log-term-grads" in text and "log_loss_term_grad_norms" in text
    assert "patch 0002 out of date" in text


# --- shot one is a 20-70 hour run, so it must survive the pod ----------------

def test_launcher_can_resume_from_the_last_checkpoint():
    """Rung 4 prices shot one at 22-68 h per arm. A run that cannot resume loses
    everything to one interruption."""
    text = LAUNCH.read_text()
    assert "--resume" in text
    assert re.search(r"resume\s*=\s*a\.resume", text), "must reach TrainConfig, not just be parsed"


def test_launcher_refuses_resume_and_overwrite_together():
    """openpi raises for this, but deep inside config validation after the model has
    started loading. Fail at the flags instead."""
    text = LAUNCH.read_text()
    assert "resume and overwrite" in text.lower()


def test_save_interval_default_is_not_never_for_long_runs():
    """The default was 10**9 -- deliberate for 10-step rungs, fatal for a 30k-step
    run, where it means one checkpoint at the very end and nothing to resume from."""
    text = LAUNCH.read_text()
    assert "--save-interval" in text
    assert "10**9" not in text or "keep_period" in text or "SAVE_INTERVAL_WARN" in text, (
        "a never-save default must at least warn when the run is long")


def test_launcher_exposes_the_seed_it_trains_with():
    """TrainConfig.seed defaults to 42. shot_one.sh records a seed in its manifest,
    and a manifest that records a seed the run never applied is worse than one that
    records none -- it is a reproduction instruction that does not reproduce."""
    text = LAUNCH.read_text()
    assert "--seed" in text
    assert re.search(r"seed\s*=\s*a\.seed", text), "must reach TrainConfig"
