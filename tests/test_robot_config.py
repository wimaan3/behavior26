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
