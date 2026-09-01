"""FIXTURE ONLY -- a stand-in for the real evaluation wrapper.

submission/build.py requires a wrapper .py to package and the organizers inspect that
file by hand to confirm the policy sees only RGB, depth and proprioception. This exists
so the packaging path is testable without a BEHAVIOR-1K checkout. It is not the real
wrapper and must never be submitted -- the real one goes in policy/wrapper.py.
"""


class MockWrapper:
    """Exposes RGB, depth and proprioception only."""

    ALLOWED_MODALITIES = ("rgb", "depth", "proprio")

    def __init__(self, env):
        self.env = env

    def observation(self, obs: dict) -> dict:
        return {k: v for k, v in obs.items() if k in self.ALLOWED_MODALITIES}
