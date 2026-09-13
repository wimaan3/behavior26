"""Does the torch thread configuration change the numbers?

WHAT THIS ANSWERS
-----------------
The worry: thread count changes float reduction order, so a "faster" thread
config could move Q -- and the organizers replicate under stock config, so a
score we cannot reproduce there gets bumped.

The causal chain is threads -> reduction order -> policy actions -> trajectory
-> Q. This cuts it at the FIRST link, for the cost of a few forward passes
rather than three rollouts: push one fixed input through the same ops under each
thread configuration and compare bitwise.

WHERE DIVERGENCE COULD ACTUALLY COME FROM
-----------------------------------------
Not the model forward. set_num_threads governs CPU intra-op parallelism; if the
policy runs on GPU its reduction order is set by the CUDA kernels regardless, so
CPU digests moving does not transfer to it. The live path is CPU-side work
BEFORE the GPU -- and in this stack that is concrete: evaluator.py:344-351 does
per-step relative_pose_transform / th.cat / mat2pose over camera poses and puts
the result in obs[...::cam_rel_poses], which reaches the policy. The
eval_small_* and eval_image_* digests below are shaped like those ops. Read
those, not cpu_matmul, when deciding whether actions can move.

WHAT IT DOES NOT ANSWER
-----------------------
Only the TORCH path. OMP_NUM_THREADS is a general OpenMP variable and other
libraries in the Isaac Sim stack may read it, so physics can still diverge even
when torch is bit-identical. That is why the rollout leg records sim_steps and
the agent_distance fields: they move continuously, so a physics-side divergence
shows up there even when a binary Q hides it. A physics-side difference is the
one that would actually break replication.

Run under each condition and diff the printed digests:
    python scripts/thread_numerics_probe.py                                  # (a) stock
    OMP_NUM_THREADS=4 python scripts/thread_numerics_probe.py                # (b)
    PYTHONPATH=scripts/threadfix BEHAVIOR_TORCH_THREADS=4 \\
        BEHAVIOR_TORCH_INTEROP=1 python scripts/thread_numerics_probe.py     # (c)
"""

from __future__ import annotations

import hashlib
import json
import sys

import torch


def digest(t: torch.Tensor) -> str:
    """Bitwise digest. Not allclose -- the question is exact reproducibility."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    torch.manual_seed(0)
    out: dict = {
        "intra_threads": torch.get_num_threads(),
        "inter_threads": torch.get_num_interop_threads(),
        "torch": torch.__version__,
    }

    # Reductions over a large-ish tensor are where thread count changes summation
    # order. Fixed seed, so the inputs are identical across conditions.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 512, generator=g, dtype=torch.float32)
    w = torch.randn(512, 512, generator=g, dtype=torch.float32)

    out["cpu_sum"] = digest(x.sum())
    out["cpu_matmul"] = digest(x @ w)
    out["cpu_mean_dim"] = digest(x.mean(dim=0))

    # A small MLP forward is the shape of a policy head.
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(512, 512), torch.nn.GELU(),
        torch.nn.Linear(512, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 23),          # R1Pro action_dim
    ).eval()
    with torch.no_grad():
        out["cpu_mlp"] = digest(net(x))

    # --- ops shaped like the ones the eval path actually runs on CPU -------------
    # evaluator.py:344-351 does per-step pose math and concatenation, and the
    # result lands in obs[...::cam_rel_poses] -- i.e. it reaches the policy. These
    # are SMALL, and torch often runs small ops single-threaded regardless of the
    # pool size, so they may be insensitive to thread count even when big matmuls
    # are not. That is the whole question, so measure it rather than assume.
    g2 = torch.Generator().manual_seed(7)
    poses = torch.randn(8, 4, 4, generator=g2, dtype=torch.float32)
    out["eval_small_matinv"] = digest(torch.linalg.inv(poses))
    out["eval_small_matmul"] = digest(poses @ poses.transpose(-1, -2))
    out["eval_small_cat"] = digest(torch.cat([poses.reshape(8, 16)] * 3, dim=-1))

    # And one image-sized op: full-res RGB+D is the large CPU tensor per step.
    img = torch.randn(3, 720, 1280, generator=g2, dtype=torch.float32)
    out["eval_image_mean"] = digest(img.mean(dim=(1, 2)))
    out["eval_image_scale"] = digest((img * 0.00392156862745098).sum())

    if torch.cuda.is_available():
        xg, wg = x.cuda(), w.cuda()
        out["cuda_sum"] = digest(xg.sum())
        out["cuda_matmul"] = digest(xg @ wg)
        with torch.no_grad():
            out["cuda_mlp"] = digest(net.cuda()(xg))
    else:
        out["cuda"] = "unavailable"

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
