# Session A handoff — 2026-09-13, unattended overnight run

Written for the next morning. Read this first, then
[`RENDERING-POSTMORTEM.md`](./RENDERING-POSTMORTEM.md) if you need the diagnosis.

---

## Start here tomorrow

```bash
# 1. Deploy a NEW pod. Same image, more vCPU than the L4 box had (6).
#    image:           ghcr.io/selkies-project/selkies-egl-desktop:26.04
#    volume:          96mu3d0s32 (EU-RO-1, 150 GB STANDARD), mounted EXPLICITLY
#                     at /workspace -- a different mount point breaks the env,
#                     it does not merely slow it down.
#    container disk:  >= 50 GB  (see "Dataset outcome" -- staging is not free)
#    Region is forced by the volume: EU-RO-1. Do not create a second volume.

# 2. On the box — NOT setup_cloud.sh. The env and dataset are already there.
source /workspace/env.sh

# 3. Assert the box can render BEFORE anything expensive (~seconds):
cd /workspace/behavior26 && bash scripts/preflight.sh

# 4. Only then: scene load, and the (a)/(b)/(c) thread matrix.
bash scripts/first_rollout.sh
```

`scripts/setup_cloud.sh` is still safe to re-run — it now detects the built env
*and* the populated dataset independently and skips both — but it has nothing
left to do.

---

## What landed tonight

| artifact | where | state |
|---|---|---|
| conda env `behavior` | `/workspace/envs/behavior` | **built, ~42 GB** |
| miniforge | `/workspace/miniforge3` | 3.8 GB |
| BEHAVIOR-1K source, tag `v3.9.2` | `/workspace/BEHAVIOR-1K` | 2.6 GB |
| `import omnigibson` | — | **succeeds** (`IMPORT_OK`, 2 m 13 s cold) |
| BEHAVIOR-1K dataset | `/workspace/og-data` | **complete, all 3 stages `EXIT=0`** |

Everything above is on network volume `96mu3d0s32` and survives pod termination.
Nothing of value is on container disk.

### The fix that unblocked the install

`selkies-egl-desktop:26.04` ships `gcc` but **not `g++`**, and `import
omnigibson` invokes TorchInductor, which needs a C++ driver. The install died
~100 minutes in, at the dataset step, with `InvalidCxxCompiler` — an error that
names neither the image nor the missing package.

Fix: `conda install -c conda-forge cxx-compiler -p /workspace/envs/behavior`.
Into the **env**, not the system — the image runs as non-root `ubuntu` with
`sudo` password-gated, so `apt` is unavailable. Now done automatically by
`setup_cloud.sh` and asserted by `preflight.sh` step 0c.

### The defect that fix exposed

`setup_cloud.sh` gated the conda env **and** the 36 GB dataset on one condition,
`if [ -d "${ENV_PREFIX}" ]`, because upstream's `setup.sh` does both in a single
invocation and hard-errors on an existing env name. So once the compiler fix made
the env exist, the re-run reported success in seconds with `og-data` still at
512 bytes — a silent partial install whose only symptom is a scene that fails to
load, later, on a box billing by the hour.

Fixed in `1593cfd`: the dataset is now `scripts/download_dataset.sh`, guarded on
the *dataset's* presence, idempotent, and runnable standalone — which is the
state a failed install actually leaves you in.

---

## Dataset outcome — COMPLETE

All three stages returned `EXIT=0`. `/workspace/og-data` holds:

```
2026-challenge-task-instances
behavior-1k-assets
omnigibson-robot-assets
omnigibson.key
```

| stage | start (UTC) | end (UTC) | wall |
|---|---|---|---|
| `download_omnigibson_robot_assets` | 07:23:45 | 07:27:57 | 4 m 12 s |
| `download_behavior_1k_assets` | 07:27:57 | 08:19:48 | **51 m 51 s** |
| `download_2026_challenge_task_instances` | 08:19:48 | 08:34:38 | 14 m 50 s |

`ALL_EXIT=0`. Pod `8aki9c2672ft6d` terminated immediately after; no pods running.

### Three things this run taught us about the download

1. **It needs ~30 GB of free CONTAINER DISK, not just volume space.** The
   `behavior_1k_assets` stage stages the whole archive in `/tmp` — container
   disk — before extracting to the volume. `/tmp/tmpm2oz7sv3` peaked at 30 GB
   against 19 GB free on a 50 GB overlay. It fit, but not by much. **Give
   tomorrow's pod a container disk of at least 50 GB**; a smaller one fails with
   ENOSPC an hour in, and the error names a temp file.
   Running each stage as its own `python -c` turned out to matter here: the temp
   dir is reclaimed when the process exits, so stage 3 started with a clean
   48 GB free. A single upstream invocation would have held all of it.
2. **The HF Hub requests are unauthenticated** — "Please set a `HF_TOKEN` to
   enable higher rate limits". That is deliberate: no token goes on rented
   hardware. It cost us nothing here (52 minutes for the bulk), but a rate-limit
   stall on a future run is not a bug to go hunting for.
3. **On-disk footprint is far larger than apparent size** — NFS block allocation
   ran about 2.5× (`du -sb` 26.1 GB vs `du -sh` 61 G, mid-transfer). The volume
   is 150 GB and the env alone is 42 GB. **First command tomorrow should be
   `du -sh /workspace/*`** — the final total was not measured, because by then
   the tree walk was outrunning its own SSH timeout and the standing instruction
   was to terminate rather than hold the box for a nice-to-have number.

---

## What did NOT happen tonight, deliberately

The (a)/(b)/(c) thread matrix, the scene-load test, and the baseline rollout were
**not run**. The L4 box has **6 vCPU against the RTX 6000 Ada's 28**, and those
numbers are CPU-bound enough that they would have to be re-measured on
representative hardware anyway. Hours of billing for a number already agreed to
discard.

---

## Why `selkies-egl-desktop:26.04`

Full table in [`RENDERING-POSTMORTEM.md`](./RENDERING-POSTMORTEM.md) — it is
already in the repo, not only in a chat log. Short version: Isaac Sim renders
through Vulkan, and NVIDIA's Vulkan ICD returns `VK_ERROR_INITIALIZATION_FAILED`
(-3) from `vk_icdNegotiateLoaderICDInterfaceVersion` inside `runpod/pytorch`
while returning **0** inside `selkies-egl-desktop`. Eight variables were held
against it across seven pods:

1. `NVIDIA_DRIVER_CAPABILITIES` — `compute,utility` vs `all`: not the cause
2. GPU device index — `/dev/nvidia0`, `nvidia2`, `nvidia6`: not the cause
3. Host driver — 580.159.03 / 580.173.02 / 595.91.07: not the cause
4. Datacenter — US-CA-2 vs EU-RO-1: not the cause
5. GPU model — RTX 6000 Ada / RTX 4090 / L4: not the cause
6. ICD manifest — host bind-mounted (`nobody:nogroup`) on **both**: not the cause
7. Vulkan loader / exported symbols — all three exported, `ldd` clean: not the cause
8. **Container image** — **the cause**

`nvidia-smi` is healthy and `torch.cuda.is_available()` is `True` on every one of
the failing pods. Nothing in the CUDA stack reveals this, which is why it cost
seven pods, and why `preflight.sh` step 0b now asserts it in seconds.

`massedcompute/nvidia-glx-desktop` also renders, but is Ubuntu 20.04 / glibc 2.31
/ python 3.8 — below the floor Isaac Sim needs. It is not a fallback.

---

## Findings worth keeping

- **Container observability is a trap here.** `nproc`, `/proc/loadavg`, `free`
  and `df` all report the **host or the storage cluster**, not our cgroup. A
  `loadavg` of 8.06 was read as ours when the container was at 0.17 of 6 cores;
  `df /workspace` reports the 2.3 PB VAST cluster, not our 150 GB quota.
  `preflight.sh` now refuses to answer the quota question rather than answer it
  wrongly.
- **NFS metadata is 14.8× slower per file** — 0.621 ms/file against 0.042 ms on
  container disk, over the env's 36,494 `.py` files (22.65 s for the walk).
  This is why plain `ls` and `stat` over SSH were timing out at 120 s tonight.
- **`import omnigibson` is ~44 s warm, 2 m 13 s cold**, and the long-standing
  "~5 minute first-import shader compile" **did not reproduce** on v3.9.2. If a
  shader compile happens it is on `og.Environment(...)`, not on import.
  Scheduling consequence recorded in `AB_PROTOCOL.md`:
  `--instances-per-job ≈ (tasks × instances)/5`.
- **The eval environment is not the organizers'** — Ubuntu 26.04 / selkies /
  driver 595.91.07. Every reported number carries an
  `scripts/env_fingerprint.py` stamp for this reason.

---

## Spend

Read from RunPod billing after termination, not estimated:

| line | amount |
|---|---|
| pod GPU | $3.97 |
| pod disk + CPU | $0.04 |
| storage — HIGH (deleted volume `rbo3wtv1po`) | $0.19 |
| storage — STANDARD (`96mu3d0s32`, ongoing) | $0.06 |
| **total to date** | **$4.26** |

Against the **$200** budget. Zero pods are running. The only ongoing cost is
volume `96mu3d0s32` at roughly **$10.50/month** — leave it alone; it holds the
env and the dataset, and rebuilding them costs far more than storing them.
