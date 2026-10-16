"""docker/policy/Dockerfile -- the submission policy must be the server that produced our results.

The public-test JSON came from scripts/eval_arms.sh -> scripts/serve_baseline.sh ->
openpi's scripts/b1k/serve_b1k.py. An image that served a different config, prompt, norm
stats or openpi commit would hand the organisers a policy we never measured.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DF = (REPO / "docker" / "policy" / "Dockerfile").read_text()


def test_the_image_pins_the_same_openpi_commit_the_runs_used():
    pinned = re.search(r'OPENPI_COMMIT="\$\{OPENPI_COMMIT:-([0-9a-f]{40})\}"',
                       (REPO / "scripts" / "install_openpi.sh").read_text()).group(1)
    assert f"ARG OPENPI_COMMIT={pinned}" in DF


def test_the_image_applies_our_patches_and_serves_arm_b_on_radio():
    assert "apply_openpi_patches.sh" in DF and "training/patches" in DF
    cmd = DF[DF.index("CMD ["):]
    for arg in ('"scripts/b1k/serve_b1k.py"', '"b1k/R1Pro"', '"b1k/turning_on_radio"',
                '"turning_on_radio"', '"8000"', '"pi05_b1k_frozen_vlm_progress"', '"/opt/checkpoint"'):
        assert arg in cmd, arg


def test_the_serve_arguments_match_what_evaluation_used():
    serve = (REPO / "scripts" / "serve_baseline.sh").read_text()
    for flag in ("--robot b1k/R1Pro", '--task "b1k/${TASK}"', '--repo-id "${ASSET_ID}"', "--policy.config", "--policy.dir"):
        assert flag in serve, flag


def test_no_optimizer_state_or_credentials_go_into_the_image():
    build = (REPO / "scripts" / "build_policy_image.sh").read_text()
    assert "train_state" not in build.split("cp -al", 1)[1].split("\n", 1)[0]
    for pattern in (r"RUNPOD_API_KEY", r"\b\w*_TOKEN\b", r"\bPASSWORD\b", r"docker login"):
        assert not re.search(pattern, DF), pattern
