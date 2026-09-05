"""
Wire format for the policy <-> evaluator websocket seam.

Both sides of the seam (``policy/null_server.py`` and ``tests/mock_evaluator.py``)
import from here so the encode/decode pair can never drift apart.

VERIFIED against BEHAVIOR-1K v3.9.2,
``omnigibson/eval/utils/network_utils.py``. Every item below is quoted from that
file; the previous version of this module guessed and got the codec wrong.

  - Health check: the client polls ``GET /healthz`` over plain HTTP every 5s until
    ``response.ok``, THEN opens the websocket (``_wait_for_server``, l.65-82).
  - Metadata frame: the client's very first action after connecting is
    ``metadata = unpackb(conn.recv())`` (l.96). It BLOCKS there. A server that does
    not send a metadata frame hangs the evaluator forever -- this is mandatory, not
    optional.
  - Observation frames: ``unpackb(await websocket.recv(), strict_map_key=False)``.
  - Reset frames: the client's ``reset()`` sends ``{"reset": True}`` and does NOT
    read a reply (l.136-141); the server answers with ``continue`` (l.187-189).
    Replying to a reset desynchronises the stream by one frame for the rest of the
    episode -- silently, since every subsequent action is still well-formed.
  - Response: ``{"action": <ndarray>, "server_timing": {...}}``. The key is
    ``action``, singular (l.119). The client then does
    ``th.from_numpy(deepcopy(action_dict["action"]))`` (l.126), so the value must
    decode to a real ``numpy.ndarray``; a Python list raises TypeError there.
  - A text frame from the server is treated as an error traceback (l.115-116).

THE CODEC IS NOT msgpack-numpy
------------------------------
network_utils.py says so explicitly: "The code below is adapted from
msgpack-numpy. The reason not to use that library directly is that it falls back
to pickle for object arrays." The formats are incompatible --

    msgpack-numpy   {b'nd': True, b'type': ..., b'kind': ..., b'shape': ..., b'data': ...}
    BEHAVIOR-1K     {b'__ndarray__': True, b'data': ..., b'dtype': ..., b'shape': ...}

so an ndarray packed by msgpack-numpy decodes on the evaluator as a plain dict,
``action_dict["action"]`` is that dict, and ``th.from_numpy`` raises. We now
implement the BEHAVIOR-1K format exactly, transcribed from ``pack_data`` /
``unpack_data``.

STILL UNVERIFIED
----------------
  - ``DEFAULT_ACTION_DIM`` (23) is read from the openpi b1k robot registry, not
    from a running evaluator.
"""

from __future__ import annotations

from typing import Any

import msgpack

# Path the evaluator health-checks before opening the websocket.
HEALTH_PATH = "/healthz"

# Action vector length for R1Pro under the b1k config. Override anywhere it is used --
# the 2026 challenge does not fix the embodiment.
DEFAULT_ACTION_DIM = 23

# The evaluator reads exactly "action" (network_utils.py l.119). We emit that; we
# still ACCEPT "actions" when decoding so a stock openpi serve script also works.
ACTION_KEYS = ("action", "actions")

# Retained for callers that used to branch on the optional msgpack-numpy import.
# The codec is now built in, so ndarray support is unconditional.
NUMPY_CODEC_AVAILABLE = True


def _pack_data(obj: Any) -> Any:
    """msgpack ``default`` hook. Transcribed from BEHAVIOR-1K ``pack_data``.

    Byte keys, and ``dtype.str`` (e.g. '<f4') rather than a name, are part of the
    format -- the evaluator's ``unpack_data`` looks for exactly ``b"__ndarray__"``.
    """
    import numpy as np  # local: keeps `import wire` cheap for non-array callers

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype for the wire: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_data(obj: Any) -> Any:
    """msgpack ``object_hook``. Transcribed from BEHAVIOR-1K ``unpack_data``."""
    import numpy as np

    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def pack(obj: Any) -> bytes:
    """Encode a message for the wire, in the evaluator's own ndarray format."""
    return msgpack.packb(obj, default=_pack_data)


def unpack(raw: bytes | str) -> Any:
    """Decode a message off the wire.

    A ``str`` means the peer sent a text frame. The evaluator uses text frames to
    carry a server traceback, so this is an error path, not an observation.
    """
    if isinstance(raw, str):
        raise TypeError(
            "expected a binary msgpack frame, got a text frame -- "
            "the evaluator sends tracebacks as text frames"
        )
    return msgpack.unpackb(raw, object_hook=_unpack_data, strict_map_key=False)


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
