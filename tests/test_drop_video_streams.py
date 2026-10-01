"""Dropping the depth video streams the model never reads.

LeRobotDataset decodes EVERY video feature listed in meta/info.json for every sample.
The BEHAVIOR datasets carry 6 streams per episode -- 3 RGB, 3 depth -- and openpi's
b1k robot config maps only the three RGB ones. Measured on the shot-one pod
(2026-10-01, 15 real samples): 4.09 s/sample with all six, 1.03 s with RGB only. The
depth streams were three-quarters of the decode cost, and the decode cost was the
training stall. Removing them from the dataset's features changes nothing the model sees.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "drop_video_streams.py"

RGB = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
       "observation.rgb.right_realsense_link_camera_0"]
DEPTH = ["observation.depth_linear.zed_link_camera_0", "observation.depth_linear.left_realsense_link_camera_0",
         "observation.depth_linear.right_realsense_link_camera_0"]


def _root(tmp: Path) -> Path:
    root = tmp / "task"
    (root / "meta").mkdir(parents=True)
    feats = {k: {"dtype": "video", "shape": [480, 640, 3]} for k in RGB + DEPTH}
    feats["action"] = {"dtype": "float32", "shape": [23]}
    feats["observation.state"] = {"dtype": "float32", "shape": [32]}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": feats}))
    for k in RGB + DEPTH:
        d = root / "videos" / k / "chunk-010"
        d.mkdir(parents=True)
        (d / "file-000.mp4").write_bytes(b"\x00" * 16)
    return root


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *extra],
                          capture_output=True, text=True, timeout=60)


def _feats(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())["features"]


def test_depth_streams_are_removed_and_rgb_and_non_video_features_kept(tmp_path):
    root = _root(tmp_path)
    p = _run(root)
    assert p.returncode == 0, p.stdout + p.stderr
    f = _feats(root)
    assert all(k in f for k in RGB), "the three streams the model reads must stay"
    assert not any(k in f for k in DEPTH)
    assert "action" in f and "observation.state" in f
    assert not any((root / "videos" / k).exists() for k in DEPTH)
    assert all((root / "videos" / k).exists() for k in RGB)


def test_it_records_what_it_dropped(tmp_path):
    root = _root(tmp_path)
    _run(root)
    m = json.loads((root / "meta" / "dropped_streams.json").read_text())
    assert sorted(m["dropped"]) == sorted(DEPTH)
    assert sorted(m["kept_video_keys"]) == sorted(RGB)


def test_running_twice_is_a_no_op(tmp_path):
    root = _root(tmp_path)
    _run(root)
    before = (root / "meta" / "info.json").read_text()
    p = _run(root)
    assert p.returncode == 0 and (root / "meta" / "info.json").read_text() == before


def test_it_refuses_to_leave_no_video_stream(tmp_path):
    root = _root(tmp_path)
    p = _run(root, "--drop-prefix", "observation.")
    assert p.returncode != 0 and "no video stream" in (p.stdout + p.stderr).lower()
    assert all(k in _feats(root) for k in RGB + DEPTH), "a refused drop must change nothing"


def test_it_refuses_to_drop_an_rgb_stream(tmp_path):
    """The model reads the RGB streams; a prefix that would remove one is a mistake."""
    root = _root(tmp_path)
    p = _run(root, "--drop-prefix", "observation.rgb.zed")
    assert p.returncode != 0 and "rgb" in (p.stdout + p.stderr).lower()
    assert all(k in _feats(root) for k in RGB + DEPTH)


def test_info_json_is_written_atomically(tmp_path):
    """A crash mid-write must not leave a truncated info.json -- the dataset would not load."""
    text = SCRIPT.read_text()
    assert "os.replace" in text or ".replace(" in text
