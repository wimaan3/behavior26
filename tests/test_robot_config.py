"""The robot's name must agree across the two files that both encode it.

Upstream ships them inconsistent:

    OmniGibson/omnigibson/eval/r1pro.yaml        name: robot_r1
    openpi/src/openpi/configs/robots/b1k.py      name="robot"      (upstream)

and every observation key is prefixed with the robot's scene name, so openpi's
serving wrapper asks for `robot::proprio` while the evaluator publishes
`robot_r1::proprio`:

    openpi/src/openpi/shared/eval_b1k_wrapper.py:72
        prop_state = obs[f"{self.robot.name}::proprio"]

That is a KeyError on the first step of the first rollout -- after the scene
load, on rented hardware. Nothing upstream checks the two files against each
other, so this test does.

It reads the actual files rather than restating their contents, so it keeps
working when either side is updated and fails the moment they disagree.

    pytest tests/test_robot_config.py -q
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# Our vendored copy, which is the artifact we submit.
OUR_ROBOT_YAML = REPO / "configs" / "robot" / "r1pro.yaml"

# The evaluator's own copy, if the checkout is present.
EVAL_ROBOT_YAML = (
    REPO.parent / "external" / "b1k" / "OmniGibson" / "omnigibson" / "eval" / "r1pro.yaml"
)

CAMERA_LINKS = ("zed_link", "left_realsense_link", "right_realsense_link")


def _yaml_robot_name(path: pathlib.Path) -> str:
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(path.read_text())
    name = cfg.get("name")
    assert name, f"{path} has no top-level `name:`"
    return name


@pytest.fixture(scope="module")
def openpi_robot():
    """The R1Pro entry from openpi's robot registry."""
    from tests.openpi_fixture import require_openpi

    require_openpi()  # skips cleanly if the fork or its deps are absent
    from openpi.configs.robots.b1k import R1Pro  # noqa: PLC0415

    return R1Pro


def test_our_robot_yaml_exists():
    """It is a required submission artifact."""
    assert OUR_ROBOT_YAML.is_file(), (
        f"{OUR_ROBOT_YAML} is missing. Copy it from "
        "OmniGibson/omnigibson/eval/r1pro.yaml in the BEHAVIOR-1K checkout."
    )


def test_robot_name_matches_between_openpi_and_the_robot_config(openpi_robot):
    """The bug this file exists for."""
    yaml_name = _yaml_robot_name(OUR_ROBOT_YAML)
    assert openpi_robot.name == yaml_name, (
        f"robot name mismatch: openpi b1k.py has {openpi_robot.name!r}, "
        f"{OUR_ROBOT_YAML.name} has {yaml_name!r}.\n"
        f"Every observation key is prefixed with this, so eval_b1k_wrapper will "
        f"look up '{openpi_robot.name}::proprio' while the evaluator publishes "
        f"'{yaml_name}::proprio' -- KeyError on the first step."
    )


def test_camera_obs_keys_carry_the_same_name(openpi_robot):
    """A stale literal in one of the three obs_keys is the same bug, quieter."""
    yaml_name = _yaml_robot_name(OUR_ROBOT_YAML)
    for camera_key, obs in openpi_robot.observations.items():
        assert obs.obs_key.startswith(f"{yaml_name}::"), (
            f"{camera_key}: obs_key {obs.obs_key!r} is not prefixed with "
            f"'{yaml_name}::'"
        )
        # OmniGibson publishes `{robot}::{robot}:{link}:Camera:0::rgb`.
        assert f"::{yaml_name}:" in obs.obs_key, (
            f"{camera_key}: obs_key {obs.obs_key!r} does not carry the robot name "
            "in the sensor path"
        )


def test_obs_keys_cover_the_cameras_the_robot_config_declares(openpi_robot):
    """The links in the yaml's eval.camera_sensor_names must be the ones openpi reads."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(OUR_ROBOT_YAML.read_text())
    sensors = (cfg.get("eval") or {}).get("camera_sensor_names") or {}
    assert sensors, f"{OUR_ROBOT_YAML} has no eval.camera_sensor_names block"

    # e.g. "robot_r1:zed_link:Camera:0" -> "zed_link"
    yaml_links = {re.split(r":", v)[1] for v in sensors.values()}
    openpi_links = {re.split(r":", obs.obs_key.split("::")[1])[1] for obs in openpi_robot.observations.values()}

    assert yaml_links == openpi_links, (
        f"camera links differ: robot config has {sorted(yaml_links)}, "
        f"openpi reads {sorted(openpi_links)}"
    )


@pytest.mark.skipif(not EVAL_ROBOT_YAML.is_file(), reason="BEHAVIOR-1K checkout not present")
def test_our_vendored_yaml_still_matches_the_evaluators():
    """We submit this file; keep it byte-identical to the organizers' own.

    We fixed the mismatch on the openpi side precisely so this stays true -- the
    robot config is inspected manually, and it must still work if they run it
    with their own copy.
    """
    ours = OUR_ROBOT_YAML.read_bytes().replace(b"\r\n", b"\n")
    theirs = EVAL_ROBOT_YAML.read_bytes().replace(b"\r\n", b"\n")
    assert ours == theirs, (
        f"{OUR_ROBOT_YAML} has drifted from {EVAL_ROBOT_YAML}. If upstream changed "
        "it, re-copy it and re-check the robot name; do not hand-edit our copy."
    )


# ----------------------------------- the compatibility patch must not be optional
#
# 2026-09-14: the first real-baseline run died on the first step of the first
# rollout with exactly the KeyError this module exists to prevent. Not because
# the check was missing -- preflight step 3 says out loud that its live
# openpi<->evaluator name check was skipped and that "it is the check that
# actually proves the first step will not KeyError" -- but because the fix lived
# inside a patch named for the progress head, and the run deliberately skipped
# that patch to keep the baseline model stock. There was no way to express
# "stock model, compatible plumbing", so nothing got applied.

from pathlib import Path as _Path  # noqa: E402

PATCH_DIR = REPO / "training" / "patches"
COMPAT = PATCH_DIR / "0001-robot-name-compatibility.patch"
CONTRIB = PATCH_DIR / "0002-progress-head-and-low-memory-configs.patch"
APPLY = REPO / "scripts" / "apply_openpi_patches.sh"


def test_the_compatibility_fix_is_its_own_patch():
    """Separable from the contribution, because they answer to different
    questions: one makes ANY run work against this evaluator, the other is what
    we are testing."""
    assert COMPAT.exists(), "the robot-name fix must be its own patch"
    files = [l for l in COMPAT.read_text().splitlines() if l.startswith("diff --git")]
    assert len(files) == 1, f"compat patch must touch exactly one file, touches {len(files)}"
    assert "configs/robots/b1k.py" in files[0]
    assert 'ROBOT_NAME = "robot_r1"' in COMPAT.read_text()


def test_the_progress_head_does_not_carry_the_robot_name_fix():
    """If it did, skipping the head would silently take the compatibility fix
    with it -- which is precisely what happened."""
    assert CONTRIB.exists()
    assert "configs/robots/b1k.py" not in CONTRIB.read_text()


def test_compat_only_is_a_first_class_mode():
    """The mode whose absence caused the failure. Running the released baseline
    wants the stock model AND working plumbing."""
    text = APPLY.read_text()
    assert "--compat-only" in text, "there must be a way to ask for stock model + compat"


def test_the_contribution_can_never_be_applied_without_the_compatibility_fix():
    """The guarantee that makes this unreasonable-out-of. Applying the progress
    head alone would reintroduce the KeyError on any evaluator run."""
    text = APPLY.read_text()
    assert "COMPAT_PATCHES" in text and "CONTRIB_PATCHES" in text, (
        "the two classes must be distinguishable in the script"
    )
    # compat must be listed before contribution wherever the apply order is built
    assert text.index("COMPAT_PATCHES") < text.index("CONTRIB_PATCHES")


def test_jax_preallocation_is_opposite_for_serving_and_training():
    """Measured 2026-09-14, and the two must not be copied to each other.

    Serving shares a 24 GB card with Isaac Sim (~14 GB), so JAX's default 75%
    preallocation starves the simulator and reports a renderer error that never
    mentions JAX. Training owns the card, where preallocation is faster and
    avoids fragmentation.
    """
    serve = (REPO / "scripts" / "serve_baseline.sh").read_text()
    train = (REPO / "scripts" / "train_cloud.sh").read_text()
    # Match the ASSIGNMENT, not the first mention -- both files discuss the flag
    # in comments before setting it.
    serve_set = re.search(r'^export XLA_PYTHON_CLIENT_PREALLOCATE=.*$', serve, re.M)
    assert serve_set and "false" in serve_set.group(0), (
        f"serving must disable preallocation; got {serve_set and serve_set.group(0)}"
    )
    train_set = re.search(r'^:\s*"\$\{XLA_PYTHON_CLIENT_PREALLOCATE:=(\w+)\}"', train, re.M)
    assert train_set, "training must state its choice rather than inherit a copied default"
    assert train_set.group(1) == "true", "training keeps preallocation ON, deliberately"
    for text in (serve, train):
        assert "Isaac Sim" in text, "each side must say why the other differs"


def test_serve_discovers_the_checkpoint_and_puts_uv_on_path():
    """Two failures from one launch, 2026-09-14.

    `exec: uv: not found` -- install_openpi.sh exports ~/.local/bin only inside
    its own shell, so any later shell loses uv.

    And the checkpoint path was hardcoded to a directory that only existed
    because the first run was unzipped by hand; install_openpi.sh puts it
    elsewhere. The directory name comes from inside the challenge's zip and is
    not ours to depend on -- find the dir containing params/ instead.
    """
    serve = (REPO / "scripts" / "serve_baseline.sh").read_text()
    assert ".local/bin" in serve, "uv must be on PATH for a fresh shell"
    assert "-name params" in serve, "the checkpoint must be discovered, not hardcoded"
    assert "/opt/baseline/extracted/" not in serve, "hardcoded path was wrong and is gone"


def test_serve_can_serve_our_own_arms_not_only_the_released_baseline():
    """Deliverable 11 evaluates arm A and arm B, which train under
    pi05_b1k_frozen_vlm -- and arm B's checkpoint carries a progress head that only
    the patched config declares. A hardcoded --policy.config pi05_b1k cannot load
    either, and the failure lands after Isaac Sim has already paid its scene load.
    """
    serve = (REPO / "scripts" / "serve_baseline.sh").read_text()
    assert re.search(r'^CONFIG="\$\{CONFIG:-', serve, re.M), "the policy config must be overridable"
    assert "--policy.config \"${CONFIG}\"" in serve
    assert "pi05_b1k_frozen_vlm" in serve, "say which config our own arms need"


def test_serve_refuses_a_checkpoint_with_no_norm_stats_in_it():
    """openpi loads norm stats from checkpoint_dir/assets/<asset_id>, NOT from the
    config's assets dir -- deliberately, so a served policy uses the statistics it
    trained with. There is no flag to point elsewhere (Checkpoint has only config
    and dir).

    The failure mode that matters: when they are absent openpi logs "not found ...
    skipping" and serves an UNNORMALISED policy, which scores near zero. On an arm,
    that reads as the treatment failing rather than as a missing file -- and it is
    discovered only after Isaac Sim has paid a 12-minute scene load."""
    serve = (REPO / "scripts" / "serve_baseline.sh").read_text()
    assert "assets" in serve and "norm_stats" in serve, "check the stats are there before serving"
    assert "--policy.assets_base_dir" not in serve, "no such flag exists on Checkpoint"
