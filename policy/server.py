"""
Websocket policy server -- the seam between our model and the official evaluator.

CONTRACT -- VERIFIED against BEHAVIOR-1K v3.9.2,
``omnigibson/eval/utils/network_utils.py``. See ``policy/wire.py`` for the frame
formats; this file only records what a POLICY object must provide.

Two ways to serve, and they want different return types:

  A. Use the official server (recommended -- it is the one the evaluator was
     written against):

         from omnigibson.eval.utils.network_utils import WebsocketPolicyServer
         WebsocketPolicyServer(policy, host=..., port=..., metadata=...).serve_forever()

     Its handler does (l.195-200):

         action = self._policy.act(obs)
         action = {"action": action.cpu().numpy()}

     so ``act`` must return a **torch.Tensor**. A numpy array has no ``.cpu()``
     and raises AttributeError on the first step. It also calls
     ``self._policy.reset()`` on a ``{"reset": True}`` frame and sends nothing
     back, so ``reset`` must exist.

  B. Use our own ``policy/null_server.py``-style server, which packs numpy
     directly. There ``act`` returns an ``np.ndarray``.

     Whichever you pick, the value that reaches the wire must decode to a real
     ndarray: the client does ``th.from_numpy(deepcopy(action_dict["action"]))``
     (l.126), which raises TypeError on a Python list.

The action width must equal ``robot.action_dim`` -- 23 for R1Pro under the b1k
config. Note ``Pi0Config.action_dim`` defaults to 32 and ``B1KOutputs`` slices
the first 23 on the way out, so a model tensor is wider than the wire action.

Everything that speaks this protocol is drop-in swappable -- baseline, our model,
a scripted policy, a random-action sanity check. That interchangeability is what
makes ablations cheap, so keep this boundary clean.

STATUS: skeleton. The protocol is now verified; what remains is model loading and
``_infer_chunk``. Until those exist, serve the baselines with the vendor scripts:
  pi0.5 :  uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/$TASK \
             --policy.config pi05_b1k --policy.dir $CKPT --port 8000
  GR00T :  python scripts/b1k/serve_b1k.py --model-path $CKPT \
             --modality-config-path examples/b1k/r1pro.py --port 8000

For a GPU-free protocol check, run ``python -m policy.null_server`` instead.
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

        Return type depends on which server wraps this policy; see the module
        docstring. Under the official WebsocketPolicyServer it must be a
        torch.Tensor of width ``robot.action_dim``, because the handler calls
        ``.cpu().numpy()`` on it.
        """
        if not self._chunk:
            chunk = self._infer_chunk(obs)
            # A chunk is (horizon, action_dim). Iterating a tensor or ndarray
            # yields per-step rows of the same type, which is what we want --
            # do not convert to Python lists, they fail th.from_numpy downstream.
            self._chunk = list(chunk)
        return self._chunk.pop(0)

    def _infer_chunk(self, obs: dict):
        """Run the model and return a (horizon, action_dim) chunk.

        TODO: load and call the model. Reuse the vendor loader rather than
        reimplementing it. Must return torch.Tensor rows under server option A.
        """
        raise NotImplementedError("TODO: run the model and return a chunk of actions")


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve a policy over websocket")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    policy = Policy(checkpoint=args.checkpoint)  # noqa: F841

    # Verified signature (network_utils.py l.150-157):
    #     WebsocketPolicyServer(policy, host="0.0.0.0", port=8000, metadata=None)
    #     .serve_forever()
    # Wiring is a two-liner once Policy._infer_chunk exists.
    raise SystemExit(
        "Not implemented: Policy._infer_chunk has no model behind it yet.\n"
        "The protocol is verified (see the module docstring); what is missing is\n"
        "checkpoint loading. Until then:\n"
        "  - baselines      -> the vendor serve scripts in the module docstring\n"
        "  - protocol check -> python -m policy.null_server"
    )


if __name__ == "__main__":
    raise SystemExit(main())
