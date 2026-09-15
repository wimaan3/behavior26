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
