"""Set torch's thread pools at interpreter startup, without editing any evaluation script.

WHY THIS EXISTS
---------------
omnigibson/eval/evaluator.py (v3.9.2) ships the remains of a known throughput fix:

    TORCH_NUM_THREADS = None
    TORCH_NUM_INTEROP_THREADS = None
    if TORCH_NUM_THREADS is not None:
        th.set_num_threads(TORCH_NUM_THREADS)

Both are None, so neither call happens and torch sizes BOTH pools from detected
cores. On a container that is the HOST's core count, not the cgroup quota --
measured 192/192 against a 20.4-core quota, ~9.4x oversubscribed on two pools at
once. A participant traced 3 fps to contact-cache thread thrashing and reported
20-30 fps with set_num_threads(4) + set_num_interop_threads(1); it shipped
disabled because it made no difference on the organizer's machine, which was
presumably not 192 cores against a 20-core quota.

WHY NOT OMP_NUM_THREADS ALONE
-----------------------------
OMP_NUM_THREADS sets the INTRA-op pool only. Measured: with OMP_NUM_THREADS=4,
intra-op is 4 and inter-op is still 192. torch exposes no environment variable
for the inter-op pool -- it has to be set from Python, before any parallel work
starts. Hence a sitecustomize, which the interpreter imports at startup.

WHY NOT EDIT THE CONSTANTS
--------------------------
Editing them is sanctioned by the organizers (they exist to be toggled), but this
touches no evaluation script at all, so there is no question to answer about the
no-modification rule. It is pure configuration.

USAGE
-----
    PYTHONPATH=scripts/threadfix BEHAVIOR_TORCH_THREADS=4 BEHAVIOR_TORCH_INTEROP=1 python ...

Inert unless those variables are set, so putting it on PYTHONPATH permanently
cannot change behaviour by accident.
"""

import os

_intra = os.environ.get("BEHAVIOR_TORCH_THREADS")
_inter = os.environ.get("BEHAVIOR_TORCH_INTEROP")

if _intra or _inter:
    try:
        import torch

        if _intra:
            torch.set_num_threads(int(_intra))
        if _inter:
            # Must happen before any inter-op parallel work; at interpreter
            # startup it always is.
            torch.set_num_interop_threads(int(_inter))
    except Exception as exc:  # never break the interpreter over a tuning knob
        print(f"sitecustomize: torch thread setup skipped: {exc}")
