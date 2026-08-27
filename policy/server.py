"""
Websocket policy server -- the seam between our model and the official evaluator.

CONTRACT (from the challenge docs; verify against your BEHAVIOR-1K checkout before use):

  - The evaluator connects to host:port and waits for a health check at /healthz,
    then opens a websocket.
  - It sends flattened observations each step.
  - We return a msgpack-encoded response containing an `action` array for the step.
  - The action vector length must equal `robot.action_dim` for the robot config in use.
  - Helper server implementation:  omnigibson/eval/utils/network_utils.py :: WebsocketPolicyServer
  - Evaluator-side client:         omnigibson.eval.policies.WebsocketPolicy

Everything that speaks this protocol is drop-in swappable -- baseline, our model, a
scripted policy, a random-action sanity check. That interchangeability is what makes
ablations cheap, so keep this boundary clean.

STATUS: skeleton. The exact import paths and message schema are NOT yet verified against
the repo -- do that in Week 1 and replace the TODOs. Do not trust this file until then.

For the baselines, use the vendor servers instead of this one:
  pi0.5 :  uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/$TASK \
             --policy.config pi05_b1k --policy.dir $CKPT --port 8000
  GR00T :  python scripts/b1k/serve_b1k.py --model-path $CKPT \
             --modality-config-path examples/b1k/r1pro.py --port 8000
"""

from __future__ import annotations

import argparse


class Policy:
    """Wraps a checkpoint and turns observations into actions."""

    def __init__(self, checkpoint: str | None = None, action_dim: int | None = None):
        self.checkpoint = checkpoint
        self.action_dim = action_dim
        # TODO: load the model. Reuse the vendor loader rather than reimplementing it.
        self.model = None

        # Action chunking: pi0.5 trains with a 32-step horizon and is served at 16.
        # We hold a chunk and only re-infer when it runs out, so any predicate/progress
        # head updates its belief ~2x/second, not 30x.
        self._chunk: list = []

    def reset(self) -> None:
        self._chunk = []

    def act(self, obs: dict):
        """Return the action for this timestep.

        `obs` carries RGB, depth and proprioception only -- that is the whole legal
        observation set. Anything derived (segmentation, point clouds, odometry) must be
        computed here from those inputs, never queried from the simulator.
        """
        if not self._chunk:
            self._chunk = list(self._infer_chunk(obs))
        return self._chunk.pop(0)

    def _infer_chunk(self, obs: dict):
        raise NotImplementedError("TODO: run the model and return a chunk of actions")


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve a policy over websocket")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    policy = Policy(checkpoint=args.checkpoint)  # noqa: F841

    raise SystemExit(
        "Not implemented.\n"
        "Week 1: import WebsocketPolicyServer from omnigibson.eval.utils.network_utils, "
        "confirm its constructor signature, and wire `policy.act` to it.\n"
        "Until then, serve the baselines with the vendor scripts in the module docstring."
    )


if __name__ == "__main__":
    raise SystemExit(main())
