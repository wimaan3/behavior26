# Session A, 2026-09-13: seven pods to find one that could render

## The corrected diagnosis

**A `BLOCKER.md` written during the session blamed `NVIDIA_DRIVER_CAPABILITIES`.
That was wrong** and is corrected here rather than preserved — a wrong diagnosis
left lying around is worse than none, because the next person acts on it.

The real cause: **the container image**. Isaac Sim renders through Vulkan, and
NVIDIA's Vulkan ICD returns `VK_ERROR_INITIALIZATION_FAILED` (-3) from
`vk_icdNegotiateLoaderICDInterfaceVersion` inside `runpod/pytorch`, while
returning 0 inside `selkies-egl-desktop` — same host driver, same datacenter,
same GPU model, same capabilities. Nothing in the CUDA stack reveals it:
`nvidia-smi` is healthy and `torch.cuda.is_available()` is `True` throughout.

## What was eliminated, and how

| variable | values tried | verdict |
|---|---|---|
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` / `all` | not the cause |
| GPU device index | `/dev/nvidia0`, `nvidia2`, `nvidia6` | not the cause |
| host driver | 580.159.03, 580.173.02, 595.91.07 | not the cause |
| datacenter | US-CA-2, EU-RO-1 | not the cause |
| GPU model | RTX 6000 Ada, RTX 4090, L4 | not the cause |
| ICD manifest | host bind-mounted (`nobody:nogroup`) on BOTH | not the cause |
| loader / exported symbols | all three ICD symbols exported, `ldd` clean | not the cause |
| **container image** | `runpod/pytorch` vs `selkies-egl-desktop` | **THE CAUSE** |

Two hypotheses were tested and disproved, which is why the list above is
trustworthy:

- *"The missing `/dev/nvidia0` breaks the ICD."* A pod with `/dev/nvidia0`
  present failed identically; a pod with `/dev/nvidia6` works.
- *"The image ships a stale ICD manifest that shadows the host's."* The manifest
  is owner `nobody:nogroup` on the failing image — bind-mounted from the host,
  not shipped. Removing it would change nothing.

## Images

| image | Vulkan | OS | usable |
|---|---|---|---|
| `runpod/pytorch:...-ubuntu2404` | **rc=-3, fails** | 24.04, glibc 2.39 | no — cannot render |
| `massedcompute/nvidia-glx-desktop` | works | **20.04, glibc 2.31, py3.8** | no — below the 22.04 floor Isaac Sim needs |
| **`ghcr.io/selkies-project/selkies-egl-desktop:26.04`** | **rc=0** | 26.04, glibc 2.43 | **yes** |

Notes on the working image: it runs as **non-root** (`ubuntu`, uid 1000) with
`sudo` password-gated, so no system packages can be installed — neither installer
needs them. It ships no `tmux`; use `nohup`. It enumerates **llvmpipe alongside
the GPU**, so selecting the GPU must be asserted, not assumed.

## The check that replaces all of this

`scripts/vulkan_check.py`, wired into `scripts/preflight.sh` as step 0b. Two
assertions: the ICD negotiates `rc=0` and returns `vkCreateInstance`, and the
enumerated device is hardware rather than a software rasteriser. Seconds to run,
on any fresh box, before the 90-minute install.

**The software-rasteriser assertion is the subtler half.** A run that lands on
llvmpipe does not crash — it is roughly 100× too slow and produces
plausible-looking timings. That is worse than a segfault, because a segfault
cannot be mistaken for data.

## Measurements that survived

From the first (non-rendering) pod, still valid — see the 2026-09-13 entries in
`AB_PROTOCOL.md`:

- `import omnigibson`: **45.0 / 44.5 / 44.1 s** — and the documented "~5 minute
  first-import shader compile" **did not reproduce** (GPU at 0%, shader cache
  empty afterwards).
- NFS metadata walk of the env's 36,494 `.py` files: **22.65 s** (0.621 ms/file)
  against 0.042 ms/file on container disk — **14.8× slower per file**.
- Cold-read cost was **not measurable** on that pod: the env was already in page
  cache from the install (84 GB cgroup limit vs a ~20 GB env).

## Cost

Roughly **$3.00** across seven pods. The diagnostic rounds themselves were about
**$0.30**; the bulk was one 90-minute install on a box that turned out unable to
render — which is exactly what the preflight check now prevents.
