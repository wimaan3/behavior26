"""
Wire format for the policy <-> evaluator websocket seam.

Both sides of the seam (``policy/null_server.py`` and ``tests/mock_evaluator.py``)
import from here so the encode/decode pair can never drift apart. If the real
protocol turns out to differ, this is the single file to correct.

CONTRACT (from the challenge docs -- NOT verified against a BEHAVIOR-1K checkout)
--------------------------------------------------------------------------------
  - Evaluator GETs ``/healthz`` over plain HTTP and expects 2xx before connecting.
  - Evaluator then opens a websocket to the same host:port.
  - Each step it sends a msgpack-encoded observation mapping.
  - The server replies with a msgpack-encoded mapping containing an ``action`` array
    whose length equals ``robot.action_dim`` (23 for R1Pro under the b1k config).

UNVERIFIED ASSUMPTIONS -- each is a knob here rather than a hardcoded guess:

  1. *msgpack-numpy.* The referenced helper
     (``omnigibson/eval/utils/network_utils.py :: WebsocketPolicyServer``) follows the
     openpi serving pattern, which packs with ``msgpack_numpy`` so raw ndarrays survive
     the hop. We use msgpack_numpy when it is importable and fall back to plain msgpack
     otherwise. Plain msgpack CANNOT encode an ndarray, so if the evaluator sends image
     arrays and msgpack_numpy is absent, decode will fail loudly rather than silently.
  2. *Server-first metadata frame.* openpi's server sends one metadata frame immediately
     on connect, before the first observation. We do the same by default
     (``--no-send-metadata`` disables it) and the mock tolerates its absence.
  3. *Response key.* The docs say ``action``. openpi uses ``actions``. We emit ``action``
     by default (``--action-key`` overrides) and accept either when decoding.

Verify all three against the real evaluator on the first GPU run.
"""

from __future__ import annotations

from typing import Any

import msgpack

# Path the evaluator health-checks before opening the websocket.
HEALTH_PATH = "/healthz"

# Action vector length for R1Pro under the b1k config. Override anywhere it is used --
# the 2026 challenge does not fix the embodiment.
DEFAULT_ACTION_DIM = 23

# Keys we will accept when reading an action out of a response frame, in priority order.
ACTION_KEYS = ("action", "actions")


def _numpy_codec():
    """Return (packb_kwargs, unpackb_kwargs) for ndarray support, if available.

    Lazy/optional import: msgpack_numpy is not in requirements-tools.txt and is not
    needed for the null policy (which sends plain floats), but the real evaluator very
    likely needs it to send image observations.
    """
    try:
        import msgpack_numpy  # type: ignore
    except ImportError:
        return {}, {}
    return ({"default": msgpack_numpy.encode}, {"object_hook": msgpack_numpy.decode})


_PACK_KW, _UNPACK_KW = _numpy_codec()

# True when ndarray-capable. Callers surface this at startup so the operator knows
# which codec is actually in play before a 20-minute rollout, not after.
NUMPY_CODEC_AVAILABLE = bool(_PACK_KW)


def pack(obj: Any) -> bytes:
    """Encode a message for the wire."""
    return msgpack.packb(obj, use_bin_type=True, **_PACK_KW)


def unpack(raw: bytes | str) -> Any:
    """Decode a message off the wire.

    Accepts ``str`` as well because a websocket peer may send a text frame; msgpack is
    binary, so a text frame means someone is speaking a different dialect and we want a
    clear error rather than a mangled observation.
    """
    if isinstance(raw, str):
        raise TypeError(
            "expected a binary msgpack frame, got a text frame -- "
            "the peer is not speaking the documented protocol"
        )
    return msgpack.unpackb(raw, raw=False, strict_map_key=False, **_UNPACK_KW)


def extract_action(response: Any) -> list:
    """Pull the action out of a decoded response frame.

    Tolerates the documented ``action`` key, openpi's ``actions`` key, and a bare
    array response. Raises with the actual payload shape on anything else -- a wrong
    guess here is exactly the failure we want to see immediately.
    """
    if isinstance(response, dict):
        for key in ACTION_KEYS:
            if key in response:
                return _as_list(response[key])
        raise KeyError(
            f"no action in response; expected one of {ACTION_KEYS}, "
            f"got keys {sorted(map(str, response.keys()))}"
        )
    if isinstance(response, (list, tuple)):
        return _as_list(response)
    raise TypeError(f"unreadable action response of type {type(response).__name__}")


def _as_list(value: Any) -> list:
    """Normalise an ndarray / nested sequence to plain lists."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def action_length(action: list) -> int:
    """Length of a single action vector, whether flat or a chunk of shape (H, dim).

    A chunked server (pi0.5 trains at horizon 32, served at 16) returns a list of
    vectors; the per-step width is what has to match ``robot.action_dim``.
    """
    if action and isinstance(action[0], (list, tuple)):
        return len(action[0])
    return len(action)
