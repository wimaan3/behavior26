"""The rendering gate, pinned.

scripts/vulkan_check.py is the check that would have saved an evening: four pods
had working CUDA and a mounted NVIDIA driver yet could not create a Vulkan
instance, and the failure surfaced as a segfault minutes into a scene load.

Its two verdicts both have to be able to FAIL, or it is decoration:
  * no device, or only a software rasteriser  -> refuse
  * ICD negotiate rc != 0                     -> refuse
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "scripts" / "vulkan_check.py"
sys.path.insert(0, str(REPO / "scripts"))

import vulkan_check  # noqa: E402


def _run(monkeypatch, devices, negotiate):
    monkeypatch.setattr(vulkan_check, "devices", lambda: devices)
    monkeypatch.setattr(vulkan_check, "icd_negotiate", lambda: negotiate)
    return vulkan_check.main()


def test_healthy_gpu_passes(monkeypatch):
    assert _run(monkeypatch, ["NVIDIA L4"], (0, "FOUND")) == 0


def test_no_device_at_all_fails(monkeypatch):
    """What a box with no working Vulkan looks like."""
    assert _run(monkeypatch, [], (0, "FOUND")) == 1


def test_software_rasteriser_only_fails(monkeypatch):
    """The dangerous case: it would RUN, ~100x too slow, producing plausible
    timings. Worse than a crash, so it must be refused rather than warned."""
    for soft in ("llvmpipe (LLVM 21.1.8, 256 bits)", "SwiftShader Device", "lavapipe"):
        assert _run(monkeypatch, [soft], (0, "FOUND")) == 1, f"{soft} was not refused"


def test_gpu_alongside_software_passes_but_is_flagged(monkeypatch, capsys):
    """Exactly what the selkies image reports: L4 + llvmpipe. Usable, but the
    caller must assert Isaac Sim picked the GPU."""
    rc = _run(monkeypatch, ["NVIDIA L4", "llvmpipe (LLVM 21.1.8, 256 bits)"], (0, "FOUND"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "software rasteriser ALSO present" in out


def test_icd_init_failure_fails(monkeypatch, capsys):
    """rc=-3 is VK_ERROR_INITIALIZATION_FAILED -- the exact code runpod/pytorch
    returned while every library and manifest was correctly in place."""
    rc = _run(monkeypatch, ["NVIDIA L4"], (-3, "NULL"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "negotiate rc=-3" in out
    assert "Change the IMAGE" in out, "must say what actually fixes it"


def test_missing_library_warns_rather_than_fails(monkeypatch):
    """A laptop with no NVIDIA driver is not a failing GPU box."""
    assert _run(monkeypatch, ["NVIDIA L4"], ("nolib", "NULL")) == 0


def test_script_runs_standalone():
    """It must work when invoked as preflight invokes it."""
    r = subprocess.run([sys.executable, str(CHECK)], capture_output=True, text=True, timeout=180)
    assert r.returncode in (0, 1)
    assert "Vulkan" in r.stdout or "ICD" in r.stdout or "vulkaninfo" in r.stdout
