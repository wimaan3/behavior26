"""
Null policy server -- returns all-zero actions. No model, no checkpoint, no GPU.

Why this exists
---------------
Two reasons, and the second is the important one.

1. It is a real baseline. A null policy scores roughly Q=0.093 on this benchmark,
   because many BEHAVIOR tasks start with a subset of their BDDL goal predicates
   already satisfied and Q is scored on the FINAL state. Doing nothing therefore banks
   whatever was true at t=0. That is ~0.09 of the ~0.10 target tier for zero work, and
   any trained policy that scores below it is actively destroying goal state.

2. It exercises the entire pipeline without a GPU. Paired with
   ``tests/mock_evaluator.py`` it makes harness/, analysis/ and submission/ testable on
   a laptop in seconds. GPU time is the scarce resource; never rent a card to discover
   that a JSON field name is wrong.

Unlike ``policy/server.py`` this file does NOT import anything from OmniGibson. The
socket handling is deliberately self-contained so it runs standalone in a bare venv
with only ``websockets`` and ``msgpack`` installed. See ``policy/wire.py`` for the
protocol contract and the list of assumptions that are not yet verified.

Usage
-----
    python -m policy.null_server --port 8000
    python -m policy.null_server --port 8000 --action-dim 23 --action-horizon 16
    python -m policy.null_server --base-port 8000 --num-servers 4   # one per eval worker
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import logging
import signal
import sys

from policy.wire import (
    DEFAULT_ACTION_DIM,
    HEALTH_PATH,
    NUMPY_CODEC_AVAILABLE,
    action_length,
    pack,
    unpack,
)

log = logging.getLogger("null_server")


class NullPolicy:
    """Returns a zero action for every observation.

    Mirrors the shape of ``policy.server.Policy`` so the two are drop-in swappable --
    that interchangeability is the whole point of the websocket seam.
    """

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM, action_horizon: int = 1):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.steps_served = 0

    def reset(self) -> None:
        self.steps_served = 0

    def act(self, obs: dict) -> list:
        """Zero action, ignoring the observation entirely.

        A flat vector when horizon == 1, otherwise a chunk of shape
        (action_horizon, action_dim).
        """
        self.steps_served += 1
        zeros = [0.0] * self.action_dim
        if self.action_horizon == 1:
            return zeros
        return [list(zeros) for _ in range(self.action_horizon)]


class NullPolicyServer:
    """Websocket server wrapping a NullPolicy.

    Health check and websocket share one port, which is what the documented protocol
    describes (``/healthz`` then upgrade on the same host:port).
    """

    def __init__(
        self,
        policy: NullPolicy,
        host: str = "127.0.0.1",
        port: int = 8000,
        action_key: str = "action",
        send_metadata: bool = True,
        max_size: int | None = 64 * 1024 * 1024,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.action_key = action_key
        self.send_metadata = send_metadata
        self.max_size = max_size
        self.episodes = 0

    # -- health check ---------------------------------------------------------------
    def _process_request(self, connection, request):
        """Answer the pre-connect health probe; let everything else upgrade.

        Returning None from process_request lets websockets proceed with the upgrade.
        We answer any non-websocket GET, not just /healthz, because the exact probe
        path is unverified -- a 200 on the wrong path is harmless, a hang is not.
        """
        path = request.path.split("?", 1)[0]
        if path.rstrip("/") in ("", HEALTH_PATH.rstrip("/")):
            # Only short-circuit if this is NOT a websocket upgrade attempt.
            if "upgrade" not in {k.lower() for k in request.headers.keys()}:
                return connection.respond(http.HTTPStatus.OK, "ok\n")
        return None

    # -- websocket ------------------------------------------------------------------
    async def _handler(self, connection) -> None:
        self.episodes += 1
        episode = self.episodes
        self.policy.reset()
        peer = getattr(connection, "remote_address", None)
        log.info("episode %d: client connected from %s", episode, peer)

        try:
            if self.send_metadata:
                # openpi's WebsocketPolicyServer sends one metadata frame on connect,
                # before any observation. UNVERIFIED for this evaluator -- disable with
                # --no-send-metadata if the real client does not expect it.
                await connection.send(pack(self._metadata()))

            async for raw in connection:
                try:
                    obs = unpack(raw)
                except Exception as exc:
                    log.error("episode %d: undecodable observation: %s", episode, exc)
                    raise

                action = self.policy.act(obs if isinstance(obs, dict) else {})
                await connection.send(pack({self.action_key: action}))

        except Exception as exc:
            # websockets raises ConnectionClosed on a normal client hangup; that is the
            # expected end of an episode, not an error.
            name = type(exc).__name__
            if "ConnectionClosed" in name:
                log.info("episode %d: client disconnected after %d step(s)",
                         episode, self.policy.steps_served)
            else:
                log.exception("episode %d: handler failed", episode)
                raise
        finally:
            log.info("episode %d: served %d step(s)", episode, self.policy.steps_served)

    def _metadata(self) -> dict:
        return {
            "policy": "null",
            "action_dim": self.policy.action_dim,
            "action_horizon": self.policy.action_horizon,
        }

    async def serve_forever(self, ready: asyncio.Event | None = None) -> None:
        from websockets.asyncio.server import serve

        async with serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._process_request,
            # Default max_size is 1 MiB. Full-res RGB+depth observations are several MiB
            # per step, so the default would drop real frames. Raised deliberately.
            max_size=self.max_size,
            ping_interval=None,   # scene load can stall the client for 150-300s
        ):
            log.info("null policy server on ws://%s:%d  (action_dim=%d, horizon=%d)",
                     self.host, self.port, self.policy.action_dim,
                     self.policy.action_horizon)
            if ready is not None:
                ready.set()
            await asyncio.Future()  # run until cancelled


async def _run(args) -> None:
    servers = [
        NullPolicyServer(
            NullPolicy(action_dim=args.action_dim, action_horizon=args.action_horizon),
            host=args.host,
            port=args.base_port + i,
            action_key=args.action_key,
            send_metadata=args.send_metadata,
            max_size=None if args.max_size <= 0 else args.max_size,
        )
        for i in range(args.num_servers)
    ]

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, stop.set)

    tasks = [asyncio.create_task(s.serve_forever()) for s in servers]
    waiter = asyncio.create_task(stop.wait())
    done, pending = await asyncio.wait([*tasks, waiter],
                                       return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*pending, return_exceptions=True)
    # Surface a real server failure rather than exiting 0 on a crashed listener.
    for t in done:
        if t is not waiter and not t.cancelled():
            t.result()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Serve an all-zero policy over the evaluator websocket protocol")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; keep on loopback unless serving to a remote evaluator")
    ap.add_argument("--port", type=int, default=8000, help="alias for --base-port")
    ap.add_argument("--base-port", type=int, default=None,
                    help="first port; server i listens on base_port + i")
    ap.add_argument("--num-servers", type=int, default=1,
                    help="how many ports to serve, one per parallel eval worker")
    ap.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM,
                    help="must equal robot.action_dim (23 for R1Pro under b1k)")
    ap.add_argument("--action-horizon", type=int, default=1,
                    help="1 = flat action vector; >1 = return a chunk of that many steps")
    ap.add_argument("--action-key", default="action",
                    help="response key holding the action ('action' per docs, 'actions' in openpi)")
    ap.add_argument("--no-send-metadata", dest="send_metadata", action="store_false",
                    help="do not send the openpi-style metadata frame on connect")
    ap.add_argument("--max-size", type=int, default=64 * 1024 * 1024,
                    help="max websocket frame bytes; <=0 for unlimited")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.set_defaults(send_metadata=True)
    args = ap.parse_args()

    if args.base_port is None:
        args.base_port = args.port
    if args.num_servers < 1:
        print("--num-servers must be >= 1", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not NUMPY_CODEC_AVAILABLE:
        log.warning("msgpack_numpy not installed -- cannot decode ndarray observations. "
                    "Fine for the mock evaluator; install it before a real run.")

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
