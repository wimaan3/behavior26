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


# ------------------------------------------------- environment fingerprint (§3.5b)

def test_fingerprint_captures_the_rendering_stack():
    """Our stack is not the organizers'. Every reported number must carry it.

    Rendering reaches Q directly -- the rendered image IS the policy input -- so a
    Q figure without the stack that produced it cannot be distinguished from a
    replication failure.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import env_fingerprint

    fp = env_fingerprint.fingerprint()
    # the rendering-specific fields, which are the point of §3.5b
    for key in ("os", "vulkan_devices", "vulkan_icd_negotiate_rc",
                "vulkan_software_rasteriser_present"):
        assert key in fp, f"fingerprint missing rendering field {key!r}"
    # provenance: which code produced the number
    for key in ("behavior26_commit", "behavior26_dirty", "python"):
        assert key in fp, f"fingerprint missing provenance field {key!r}"
    # thread config, tying it to §3.5a
    assert "OMP_NUM_THREADS" in fp


def test_fingerprint_is_json_serialisable():
    """It gets embedded in a JSONL result row; an unserialisable value would
    lose the whole rollout record rather than just the fingerprint."""
    import json
    sys.path.insert(0, str(REPO / "scripts"))
    import env_fingerprint
    json.dumps(env_fingerprint.fingerprint(), sort_keys=True)


def test_fingerprint_survives_a_box_with_no_gpu():
    """It runs on a laptop during development too -- it must degrade, not raise."""
    sys.path.insert(0, str(REPO / "scripts"))
    import env_fingerprint
    fp = env_fingerprint.fingerprint()
    assert fp["vulkan_devices"] == [] or isinstance(fp["vulkan_devices"], list)


# ------------------------------------------------- late-failure traps (preflight 0d/0e)

import subprocess as _sp


# preflight.sh runs `pytest tests/`, so a test that runs preflight recurses
# forever. preflight exports PREFLIGHT_RUNNING=1; these skip on it. (An earlier
# note in scripts/preflight.sh said "there is deliberately no test that invokes
# it" for exactly this reason -- this is that constraint, enforced rather than
# written down.)
nested = pytest.mark.skipif(
    bool(__import__("os").environ.get("PREFLIGHT_RUNNING")),
    reason="invoked from preflight itself; running it again would recurse")


def _preflight(env_extra=None, pre=""):
    """Run preflight and return its output. `pre` runs first in the same shell."""
    import shlex
    # REPO contains a space on some checkouts -- quote it or bash splits the path.
    cmd = f"{pre}bash {shlex.quote(str(REPO / 'scripts' / 'preflight.sh'))}"
    env = dict(**{k: v for k, v in __import__("os").environ.items()}, **(env_extra or {}))
    return _sp.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, timeout=900).stdout


@nested
def test_open_file_limit_failure_fires():
    """Isaac Sim runs out of descriptors PARTWAY THROUGH scene load, and reports
    whichever file was next rather than the limit."""
    out = _preflight(pre="ulimit -Sn 256 2>/dev/null; ")
    assert "open-file limit is only 256" in out
    assert "Raise it: ulimit -n" in out, "must say how to fix it"


@nested
def test_unwritable_tmp_failure_fires():
    out = _preflight(env_extra={"TMPDIR": "/nonexistent-tmp-preflight"})
    assert "is not writable" in out


@nested
def test_wrong_python_fails_only_inside_the_behavior_env():
    """A red preflight on a dev laptop teaches people to ignore preflight, which
    would cost us the checks that matter. So: warn outside the env, fail inside."""
    outside = _preflight()
    assert "not the behavior env" in outside
    inside = _preflight(env_extra={"CONDA_DEFAULT_ENV": "behavior"})
    assert "behavior env python is" in inside
    assert "prebuilt wheels" in inside, "must explain why the wrong python is slow, not just wrong"


def test_preflight_names_the_late_failures_it_prevents():
    """Each check exists because the real failure is late and misreported. The
    message has to carry that, or the next person deletes the check."""
    text = (REPO / "scripts" / "preflight.sh").read_text()
    for phrase in ("bus error", "mid scene load", "prebuilt wheels",
                   "shared cluster, not our quota", "100x too slow"):
        assert phrase in text, f"preflight no longer explains: {phrase!r}"


def test_shm_and_jax_checks_are_present():
    text = (REPO / "scripts" / "preflight.sh").read_text()
    assert "/dev/shm" in text and "DataLoader workers" in text
    assert "0e. session B readiness (jax)" in text
    assert "quietly train on CPU" in text, "the jax failure is silent, not a crash"
