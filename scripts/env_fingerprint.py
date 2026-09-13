"""Emit the full environment fingerprint as JSON.

WHY
---
We evaluate on a stack nobody else runs -- Ubuntu 26.04 /
selkies-egl-desktop:26.04 / driver 595.91.07 -- because it is the only image
tested where NVIDIA's Vulkan ICD initialises at all. Rendering differences reach
Q directly: the rendered image IS the policy's input, so different hardware,
driver or rasteriser means different pixels, different actions, and a trajectory
that diverges further with every step.

dQ is paired and the environment is constant across arms, so it cancels and the
A/B stands. Absolute Q may not replicate on the organizers' stack, and their
protocol bumps submissions they cannot reproduce.

So every reported number carries the stack that produced it. Without it we cannot
tell a replication failure from a real effect -- which is the distinction the
whole protocol exists to protect. See AB_PROTOCOL.md §3.5b.

    python scripts/env_fingerprint.py            # pretty
    python scripts/env_fingerprint.py --compact  # one line, for embedding
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sh(*cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def fingerprint() -> dict:
    fp: dict = {}

    # --- box -------------------------------------------------------------------
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                osr[k] = v.strip('"')
    except OSError:
        pass
    fp["os"] = osr.get("PRETTY_NAME") or platform.platform()
    fp["glibc"] = (platform.libc_ver() or ("", ""))[1] or None
    fp["kernel"] = platform.release()
    fp["image"] = os.environ.get("RUNPOD_IMAGE_NAME") or os.environ.get("IMAGE") or None

    # --- gpu / driver ----------------------------------------------------------
    smi = _sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader")
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        fp["gpu"] = parts[0] if parts else None
        fp["gpu_memory"] = parts[1] if len(parts) > 1 else None
        fp["driver"] = parts[2] if len(parts) > 2 else None
    fp["nvidia_devices"] = sorted(
        p.name for p in Path("/dev").glob("nvidia*") if p.name != "nvidia-caps")
    fp["driver_capabilities"] = os.environ.get("NVIDIA_DRIVER_CAPABILITIES")

    # --- RENDERING. The part that makes this stack unlike the organizers'. ------
    vk = _sh("vulkaninfo", timeout=120)
    devs = sorted({m.group(1).strip() for m in re.finditer(r"deviceName\s*=\s*(.+)", vk)})
    fp["vulkan_devices"] = devs
    fp["vulkan_software_rasteriser_present"] = any(
        re.search(r"llvmpipe|swiftshader|lavapipe", d, re.I) for d in devs)
    try:
        import ctypes
        lib = ctypes.CDLL("libGLX_nvidia.so.0")
        v = ctypes.c_uint32(5)
        fp["vulkan_icd_negotiate_rc"] = lib.vk_icdNegotiateLoaderICDInterfaceVersion(
            ctypes.byref(v))
    except Exception:
        fp["vulkan_icd_negotiate_rc"] = None

    # --- python / torch / threads ----------------------------------------------
    fp["python"] = platform.python_version()
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda_available"] = torch.cuda.is_available()
        fp["torch_intra_threads"] = torch.get_num_threads()
        fp["torch_inter_threads"] = torch.get_num_interop_threads()
    except Exception:
        fp["torch"] = None
    fp["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")

    # --- provenance: what code produced this -----------------------------------
    fp["behavior26_commit"] = _sh("git", "-C", str(REPO), "rev-parse", "--short", "HEAD") or None
    fp["behavior26_dirty"] = bool(_sh("git", "-C", str(REPO), "status", "--porcelain"))
    b1k = Path(os.environ.get("BEHAVIOR_ROOT", "/workspace/BEHAVIOR-1K"))
    if (b1k / ".git").exists():
        fp["behavior1k_tag"] = _sh("git", "-C", str(b1k), "describe", "--tags", "--always") or None
    openpi = Path(os.environ.get("OPENPI_ROOT", REPO.parent / "openpi"))
    if (openpi / ".git").exists():
        fp["openpi_commit"] = _sh("git", "-C", str(openpi), "rev-parse", "--short", "HEAD") or None
    patches = sorted((REPO / "training" / "patches").glob("*.patch"))
    if patches:
        fp["openpi_patch_sha256"] = _sha256(patches[0])

    # --- data ------------------------------------------------------------------
    root = os.environ.get("DATASET_ROOT")
    fp["dataset_root"] = root
    if root:
        pf = Path(root) / "meta" / "progress_filter.json"
        fp["progress_filter_sha256"] = _sha256(pf)

    return fp


def main() -> int:
    fp = fingerprint()
    if "--compact" in sys.argv:
        print(json.dumps(fp, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(fp, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
