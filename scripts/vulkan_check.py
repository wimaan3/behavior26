"""Can this box actually render? Exit 0 if yes, 1 if no.

WHY THIS EXISTS
---------------
Isaac Sim renders through Vulkan. On 2026-09-13, four RunPod pods had:
working CUDA, torch.cuda.is_available() True, nvidia-smi healthy, the NVIDIA
driver bind-mounted, vk_icdGetInstanceProcAddr exported, a valid ICD manifest --
and still could not create a Vulkan instance. The driver's own ICD init returned
VK_ERROR_INITIALIZATION_FAILED (-3). Nothing in the CUDA stack reveals this.

The failure surfaces as a SEGFAULT several minutes into a scene load, after the
90-minute install and the 36 GB download are already paid for.

Eliminated as causes across seven pods: GPU device index (/dev/nvidia0 vs 2 vs 6),
NVIDIA_DRIVER_CAPABILITIES, host driver version, datacenter, GPU model, the ICD
manifest (it is host-mounted, not image-shipped), and loader configuration. The
one variable that mattered was the container IMAGE.

TWO ASSERTIONS
--------------
1. The ICD negotiates (rc=0) and returns vkCreateInstance.
2. The enumerated device is real hardware, NOT llvmpipe. A run on the software
   rasteriser does not crash -- it is merely ~100x too slow, which is worse,
   because it yields plausible-looking timings that are entirely wrong.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys

SOFTWARE = re.compile(r"llvmpipe|swiftshader|lavapipe|software", re.I)


def icd_negotiate() -> tuple[object, str]:
    """Ask the NVIDIA ICD to negotiate directly. rc=0 means it initialised."""
    try:
        lib = ctypes.CDLL("libGLX_nvidia.so.0")
    except OSError:
        return "nolib", "NULL"
    try:
        version = ctypes.c_uint32(5)
        rc = lib.vk_icdNegotiateLoaderICDInterfaceVersion(ctypes.byref(version))
        gipa = lib.vk_icdGetInstanceProcAddr
        gipa.restype = ctypes.c_void_p
        gipa.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        return rc, ("FOUND" if gipa(None, b"vkCreateInstance") else "NULL")
    except AttributeError:
        return "nosym", "NULL"


def devices() -> list[str]:
    try:
        out = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return sorted({m.group(1).strip() for m in re.finditer(r"deviceName\s*=\s*(.+)", out)})


def main() -> int:
    ok = True
    devs = devices()

    if not devs:
        print("  FAIL  vulkaninfo enumerated NO device -- this box cannot render")
        ok = False
    else:
        hardware = [d for d in devs if not SOFTWARE.search(d)]
        if not hardware:
            print(f"  FAIL  ONLY a software rasteriser available: {devs}")
            print("        A run would not crash, it would be ~100x too slow, and the")
            print("        timings would look plausible. That is worse than a crash.")
            ok = False
        else:
            print(f"  ok    Vulkan hardware device: {hardware[0]}")
            if len(devs) > len(hardware):
                soft = [d for d in devs if SOFTWARE.search(d)]
                print(f"  warn  software rasteriser ALSO present {soft} --")
                print("        assert Isaac Sim selects the GPU, do not assume it")

    rc, fn = icd_negotiate()
    if rc == 0 and fn == "FOUND":
        print("  ok    NVIDIA ICD negotiate rc=0, vkCreateInstance available")
    elif rc in ("nolib", "nosym"):
        print(f"  warn  libGLX_nvidia.so.0 not usable here ({rc}) -- non-NVIDIA box?")
    else:
        print(f"  FAIL  NVIDIA ICD negotiate rc={rc}, vkCreateInstance={fn}")
        print("        rc=-3 is VK_ERROR_INITIALIZATION_FAILED: the driver refuses to")
        print("        initialise in this container. Change the IMAGE, not the config --")
        print("        capabilities, device index and datacenter were all eliminated.")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
