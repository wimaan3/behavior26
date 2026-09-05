"""
Mock evaluator -- stands in for ``python -m omnigibson.eval.eval`` with no GPU.

Why this exists
---------------
A real rollout is 20-25 minutes and costs GPU-hours we do not have. Every bug in
harness/, analysis/ or submission/ that we find on a rented card is a bug we paid for
twice. This script speaks the same CLI and the same websocket protocol as the real
evaluator and writes rollout JSONs in the same schema, so the whole chain

    policy server -> harness/launch.py -> analysis/* -> submission/build.py

runs on a laptop in seconds. When we do rent hardware, we run things we already know
work.

What it is NOT
--------------
It does not simulate physics, does not render, and its numbers are synthetic. It proves
plumbing and schema, never behaviour. A green run here says "the pipeline is wired
correctly", not "the policy is good".

Fidelity notes (all unverified against the real evaluator -- see policy/wire.py):
  - The output filename convention is now VERIFIED against v3.9.2. If it changes, fix it here and
    in ``harness.launch.already_done``.
  - ``agent_distance`` is derived from the magnitude of the actions the policy actually
    returns, so a null (all-zero) policy correctly produces zero displacement and tags
    as IMMOBILE downstream. That makes the mock exercise analysis/failures.py for real
    rather than emitting arbitrary numbers.
  - ``q_score.final`` is synthetic, drawn around ``--target-q``. It is a knob for
    testing analysis code, not a prediction.

Usage
-----
    # same CLI as the real evaluator
    python -m tests.mock_evaluator --task-name turning_on_radio --host 127.0.0.1 \
        --port 8000 --instance-indices 0 1 2 --num-rollouts 1 \
        --output-dir rollouts/mock --write-video --headless

    # mock-only knobs
    python -m tests.mock_evaluator ... --fast --target-q 0.35 --seed 7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from policy.wire import DEFAULT_ACTION_DIM, action_length, extract_action, pack, unpack

# Organizers' published throughput for full-res RGB+depth, used to fake step pacing
# when --fast is off.
SIM_FPS = 13.5
# Per-trial scene load, 150-300s in reality. Scaled down hard even in "realistic" mode;
# nobody wants a 5-minute mock.
FAKE_SCENE_LOAD_S = 0.5


# --------------------------------------------------------------------------------------
# fake observations
# --------------------------------------------------------------------------------------
def make_observation(step: int, task: str, instance_id: int, use_images: bool) -> dict:
    """Build one plausible observation frame.

    The legal observation set is RGB, depth and proprioception only. By default we send
    small numeric stand-ins rather than real image arrays: plain msgpack cannot encode
    ndarrays, and the point here is to exercise the protocol, not the pixel bus. Pass
    --with-images to send actual array-shaped payloads (needs msgpack_numpy on both ends).
    """
    obs: dict = {
        "step": step,
        "task": task,
        "instance_id": instance_id,
        "proprio": [0.0] * DEFAULT_ACTION_DIM,
    }
    if use_images:
        try:
            import numpy as np
        except ImportError:
            raise SystemExit("--with-images needs numpy")
        # Tiny, not full-res -- we are testing the wire, not the bandwidth.
        obs["rgb"] = {
            "head": np.zeros((32, 32, 3), dtype="uint8"),
            "left_wrist": np.zeros((32, 32, 3), dtype="uint8"),
            "right_wrist": np.zeros((32, 32, 3), dtype="uint8"),
        }
        obs["depth"] = {"head": np.zeros((32, 32), dtype="float32")}
    else:
        obs["rgb_shape"] = [3, 224, 224]
        obs["depth_shape"] = [1, 224, 224]
    return obs


# --------------------------------------------------------------------------------------
# one rollout
# --------------------------------------------------------------------------------------
async def run_rollout(args, instance_id: int, rollout_id: int) -> dict:
    """Drive one episode against the policy server and return the metrics dict."""
    from websockets.asyncio.client import connect

    # Two independent random streams, deliberately.
    #
    #   difficulty -- seeded on (task, instance) ONLY, so it is IDENTICAL across runs.
    #                 Real instances differ in intrinsic hardness; that variance is most
    #                 of the total and it is exactly what paired comparison cancels.
    #   rng        -- seeded on the run seed too, so it differs between runs. This is the
    #                 evaluator's own nondeterminism.
    #
    # Without the split, two runs would be statistically independent and pairing would
    # buy nothing -- the mock would silently fail to exercise analysis/compare.py.
    difficulty = random.Random(f"difficulty:{args.task_name}:{instance_id}")
    rng = random.Random(f"{args.seed}:{args.task_name}:{instance_id}:{rollout_id}")
    instance_offset = difficulty.gauss(0.0, args.instance_variance)

    max_steps = args.max_steps if args.max_steps else args.default_max_steps
    # Episode length: either the full budget (a timeout) or an early finish.
    if args.episode_length is not None:
        steps = min(args.episode_length, max_steps)
    elif rng.random() < args.timeout_rate:
        steps = max_steps
    else:
        # Harder instances run longer, so length correlates with difficulty too.
        base = difficulty.uniform(0.3, 0.95)
        steps = max(1, int(max_steps * min(1.0, base + rng.gauss(0.0, 0.05))))

    if not args.fast:
        await asyncio.sleep(FAKE_SCENE_LOAD_S)

    uri = f"ws://{args.host}:{args.port}"
    action_energy = 0.0      # summed |action| over the episode, used for agent_distance
    base_energy = 0.0
    left_energy = 0.0
    right_energy = 0.0
    executed = 0

    async with connect(uri, max_size=None, ping_interval=None,
                       open_timeout=args.connect_timeout) as ws:
        # The server may send a metadata frame first (openpi does). Tolerate both:
        # wait briefly, and treat a timeout as "this server does not send one".
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=args.metadata_timeout)
            meta = unpack(first)
            print(f"    server metadata: {meta}")
        except asyncio.TimeoutError:
            meta = None
        except Exception as exc:
            print(f"    ! could not read metadata frame: {exc}")
            meta = None

        step_period = 0.0 if args.fast else 1.0 / SIM_FPS

        for step in range(steps):
            obs = make_observation(step, args.task_name, instance_id, args.with_images)
            await ws.send(pack(obs))

            raw = await asyncio.wait_for(ws.recv(), timeout=args.step_timeout)
            action = extract_action(unpack(raw))

            width = action_length(action)
            if width != args.action_dim:
                raise SystemExit(
                    f"action_dim mismatch: server returned {width}, "
                    f"expected {args.action_dim}. The real evaluator rejects this -- "
                    f"the vector must equal robot.action_dim exactly."
                )

            # Flatten a chunk down to its first step; the evaluator consumes one step
            # at a time regardless of how many the server hands back.
            first_step = action[0] if action and isinstance(action[0], (list, tuple)) else action
            vals = [float(v) for v in first_step]
            mag = sum(abs(v) for v in vals)
            action_energy += mag
            # Rough limb split: base is the leading dims, then left arm, then right.
            third = max(1, len(vals) // 3)
            base_energy += sum(abs(v) for v in vals[:third])
            left_energy += sum(abs(v) for v in vals[third:2 * third])
            right_energy += sum(abs(v) for v in vals[2 * third:])

            executed += 1
            if step_period:
                await asyncio.sleep(step_period)

    # ---- synthesise metrics ----------------------------------------------------------
    q = args.target_q + instance_offset + rng.gauss(0.0, args.q_jitter)
    q = min(1.0, max(0.0, q))
    if args.quantize_q > 1:
        # Q is a fraction of satisfied predicates, so real values are k/N, not continuous.
        q = round(q * args.quantize_q) / args.quantize_q

    if args.distance_mode == "actions":
        scale = args.distance_scale
        dist_base = base_energy * scale
        dist_left = left_energy * scale
        dist_right = right_energy * scale
    else:
        dist_base = rng.uniform(0.0, 6.0)
        dist_left = rng.uniform(0.0, 1.5)
        dist_right = rng.uniform(0.0, 1.5)

    sim_time = executed / SIM_FPS
    return {
        "task": args.task_name,
        "instance_id": instance_id,
        "rollout_id": rollout_id,
        "steps": executed,
        "success": bool(q >= args.success_threshold),
        "agent_distance": {
            "base": round(dist_base, 4),
            "left": round(dist_left, 4),
            "right": round(dist_right, 4),
        },
        "normalized_agent_distance": round(dist_base / max(args.distance_norm, 1e-9), 4),
        "q_score": {"final": round(q, 4)},
        "time": {
            "simulator_steps": executed,
            "simulator_time": round(sim_time, 3),
            "normalized_time": round(executed / max(max_steps, 1), 4),
        },
    }


# VERIFIED against BEHAVIOR-1K v3.9.2.
#   omnigibson/eval/utils/eval_utils.py
#       TEST_INSTANCE_IDS = list(range(301, 341))
#       NUM_PUBLIC_TEST_INSTANCES = 20
#   omnigibson/eval/evaluator.py :: resolve_instance_ids
#       public_test -> TEST_INSTANCE_IDS[:20], hidden_test -> TEST_INSTANCE_IDS[20:]
#       and asserts the indices lie in range(len(split)).
#
# So --instance-indices are INDICES INTO THE SPLIT, not instance ids: index 0 of
# public_test is instance 301. The rollout JSON and its filename carry the
# RESOLVED id (eval.py l.178, l.184).
TEST_INSTANCE_IDS = list(range(301, 341))
NUM_PUBLIC_TEST_INSTANCES = 20
EVAL_MODES = ("train", "public_test", "hidden_test")


def resolve_instance_ids(instance_indices: list[int], mode: str) -> list[int]:
    """Mirror of omnigibson.eval.evaluator.resolve_instance_ids."""
    if mode == "train":
        return [int(i) for i in instance_indices]
    split = (
        TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
        if mode == "public_test"
        else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
    )
    bad = [i for i in instance_indices if not 0 <= i < len(split)]
    if bad:
        raise SystemExit(
            f"instance indices {bad} out of range for mode {mode!r}: "
            f"must be in range({len(split)}). These are indices into the split, "
            f"not instance ids ({split[0]}-{split[-1]})."
        )
    return [int(split[i]) for i in instance_indices]


def rollout_filename(task: str, instance_id: int, rollout_id: int) -> str:
    """VERIFIED: omnigibson/eval/eval.py l.184.

        out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")

    The official scorer asserts every file in json/ matches
    f"{task}_{instance}_{rollout}.json" for a real task and a valid split
    instance (utils/score_utils.py l.298, l.315), so any other name -- and any
    stray .json such as a timing manifest -- makes it raise.
    """
    return f"{task}_{instance_id}_{rollout_id}.json"


async def run_all(args) -> int:
    json_dir = args.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    started = time.time()

    instance_ids = resolve_instance_ids(args.instance_indices, args.mode)
    if instance_ids != list(args.instance_indices):
        print(f"  [mock-eval] mode={args.mode}: indices {list(args.instance_indices)} "
              f"-> instances {instance_ids}")

    for instance_id in instance_ids:
        for rollout_id in range(args.num_rollouts):
            label = f"{args.task_name} inst={instance_id} rollout={rollout_id}"
            print(f"  [mock-eval] {label}")
            try:
                result = await run_rollout(args, instance_id, rollout_id)
            except SystemExit:
                raise
            except Exception as exc:
                print(f"  ! {label} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
                continue

            path = json_dir / rollout_filename(args.task_name, instance_id, rollout_id)
            path.write_text(json.dumps(result, indent=2))
            written += 1
            print(f"    q={result['q_score']['final']:.3f} "
                  f"steps={result['steps']} success={result['success']} -> {path.name}")

            if args.write_video:
                # The real evaluator writes an MP4 per rollout. Videos are submitted as a
                # link, never zipped, so a placeholder is enough to keep paths honest.
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                (video_dir / path.name.replace(".json", ".mp4")).write_bytes(b"")

    elapsed = time.time() - started
    print(f"  [mock-eval] {written} rollout(s) written, {failed} failed, {elapsed:.1f}s")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Mock BEHAVIOR evaluator -- same CLI and protocol, no simulator")

    # ---- the real evaluator's interface (must stay in sync with harness.launch) ------
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--instance-indices", type=int, nargs="+", default=[0],
                    help="indices into the split selected by --mode, NOT instance ids")
    ap.add_argument("--mode", choices=EVAL_MODES, default="public_test",
                    help="instance split to evaluate (real evaluator default: public_test)")
    ap.add_argument("--policy", choices=("websocket", "local"), default="websocket",
                    help="present for CLI parity; 'local' is not implemented in the mock")
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--env-wrapper", default=None)
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=None)
    ap.add_argument("--headless", dest="headless", action="store_true", default=True)
    ap.add_argument("--no-headless", dest="headless", action="store_false")

    # ---- mock-only knobs -------------------------------------------------------------
    mock = ap.add_argument_group("mock-only (not present on the real evaluator)")
    mock.add_argument("--fast", action="store_true",
                      help="skip all simulated timing; finish in seconds")
    mock.add_argument("--target-q", type=float, default=0.093,
                      help="mean q_score.final to synthesise (default: the null-policy baseline)")
    mock.add_argument("--q-jitter", type=float, default=0.05,
                      help="per-run noise: stddev of the evaluator's own nondeterminism")
    mock.add_argument("--instance-variance", type=float, default=0.06,
                      help="per-instance intrinsic difficulty stddev; SHARED across runs, "
                           "which is the variance paired comparison cancels")
    mock.add_argument("--quantize-q", type=int, default=0,
                      help="round Q to k/N for N predicates; 0 = continuous")
    mock.add_argument("--success-threshold", type=float, default=1.0,
                      help="success = q >= this (1.0 = all goal predicates satisfied)")
    mock.add_argument("--episode-length", type=int, default=None,
                      help="force every episode to this many steps")
    mock.add_argument("--default-max-steps", type=int, default=1000,
                      help="stand-in for the task-specific default when --max-steps is unset")
    mock.add_argument("--timeout-rate", type=float, default=0.3,
                      help="fraction of episodes that run to the step limit")
    mock.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM,
                      help="reject actions whose width is not this")
    mock.add_argument("--distance-mode", choices=["actions", "random"], default="actions",
                      help="derive agent_distance from action magnitude, or draw at random")
    mock.add_argument("--distance-scale", type=float, default=0.01,
                      help="metres of displacement per unit of summed action magnitude")
    mock.add_argument("--distance-norm", type=float, default=10.0,
                      help="denominator for normalized_agent_distance")
    mock.add_argument("--with-images", action="store_true",
                      help="send real ndarray image payloads (needs msgpack_numpy both ends)")
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--connect-timeout", type=float, default=10.0)
    mock.add_argument("--metadata-timeout", type=float, default=0.5)
    mock.add_argument("--step-timeout", type=float, default=30.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.num_rollouts < 1:
        print("--num-rollouts must be >= 1", file=sys.stderr)
        return 2
    print(f"[mock-eval] {args.task_name} -> ws://{args.host}:{args.port} "
          f"instances={args.instance_indices} rollouts={args.num_rollouts}")
    try:
        return asyncio.run(run_all(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
