"""Estimate the within-instance noise floor sigma_w from repeated rollouts.

AB_PROTOCOL section 2. The A/B's minimum detectable effect depends on sigma_w, and
until it is measured every MDE in the design table is a hypothesis. The decision it
must serve is coarse: **is f below 0.10 or above 0.20** -- the difference between 1
seed and 3 seeds per instance, and 3x the evaluation budget.

Design (fixed in the protocol, not chosen here): 12 TRAINING instances of
turning_on_radio, r = 6 repeats each, the released baseline policy -- not the null
policy, whose within-instance variance is trivially zero and would report sigma_w = 0.

Estimator, per the protocol:

    per instance:  s2_i = r/(r-1) * p_i * (1 - p_i)     # unbiased for pi(1-pi)
    pooled:        s2_w = mean_i s2_i    ->    sigma_w = sqrt(s2_w)

`q` is used rather than `success` when the task is graded; on a D=1 task they are the
same number. For graded tasks the same formula is applied to the per-instance variance
of Q directly, which is what enters the power model.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoiseFloor:
    n_instances: int
    repeats: list[int]
    sigma_w: float
    per_instance_var: dict[int, float]
    p_hat: dict[int, float]
    n_flipped: int          # instances with at least one disagreeing repeat
    binary: bool
    mean_q: float

    @property
    def f_implied(self) -> float | None:
        """For a binary (D=1) task, sigma_w = 0.5*sqrt(f) at step=1, so f = 4*sigma_w^2."""
        return 4 * self.sigma_w ** 2 if self.binary else None


def load_rollouts(json_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(json_dir).glob("*.json")):
        d = json.loads(p.read_text())
        q = d.get("q_score")
        out.append({
            "instance_id": int(d["instance_id"]),
            "rollout_id": int(d.get("rollout_id", 0)),
            "q": float(q["final"] if isinstance(q, dict) else q),
            "success": bool(d.get("success", False)),
        })
    return out


def estimate(rollouts: list[dict]) -> NoiseFloor:
    if not rollouts:
        raise ValueError("no rollouts")
    by_inst: dict[int, list[float]] = {}
    for r in rollouts:
        by_inst.setdefault(r["instance_id"], []).append(r["q"])
    singles = [i for i, v in by_inst.items() if len(v) < 2]
    if singles:
        raise ValueError(f"instances {sorted(singles)} have a single rollout -- sigma_w needs repeats "
                         f"(the protocol specifies r=6); one rollout per instance measures nothing")
    values = [q for v in by_inst.values() for q in v]
    binary = set(values) <= {0.0, 1.0}
    per_var, p_hat = {}, {}
    for i, v in by_inst.items():
        r = len(v)
        p = statistics.fmean(v)
        p_hat[i] = p
        # r/(r-1) * p(1-p) is the unbiased estimate of pi(1-pi) for a binary outcome;
        # for graded Q the same correction gives the unbiased sample variance.
        per_var[i] = (r / (r - 1)) * p * (1 - p) if binary else statistics.variance(v)
    flipped = sum(1 for v in by_inst.values() if len(set(v)) > 1)
    return NoiseFloor(
        n_instances=len(by_inst),
        repeats=sorted(len(v) for v in by_inst.values()),
        sigma_w=math.sqrt(statistics.fmean(per_var.values())),
        per_instance_var=per_var, p_hat=p_hat, n_flipped=flipped, binary=binary,
        mean_q=statistics.fmean(values),
    )


def report(nf: NoiseFloor) -> str:
    lines = [
        f"instances            {nf.n_instances} (repeats {nf.repeats[0]}-{nf.repeats[-1]})",
        f"mean Q               {nf.mean_q:.4f}",
        f"outcome              {'binary' if nf.binary else 'graded'}",
        f"instances that flipped at least once  {nf.n_flipped}/{nf.n_instances}",
        f"SIGMA_W              {nf.sigma_w:.4f}",
    ]
    if nf.f_implied is not None:
        f = nf.f_implied
        lines.append(f"implied f            {f:.3f}")
        if f < 0.10:
            verdict = "f < 0.10 -> 1 seed per instance is the right design"
        elif f > 0.20:
            verdict = "f > 0.20 -> 3 seeds per instance; budget accordingly"
        else:
            verdict = "f in [0.10, 0.20] -> the coarse question is unresolved; see AB_PROTOCOL section 2"
        lines.append(f"decision             {verdict}")
        if f > 0.4:
            lines.append("WARNING: f > 0.4 -- a per-instance binary A/B is not viable at this budget; "
                         "the protocol's fallback is a graded task (cook_bacon)")
    if nf.n_flipped == 0:
        lines.append("WARNING: no instance flipped. sigma_w reads 0, but with r repeats that is also what "
                     "a deterministic-looking sample gives; treat as an upper bound, not a measurement.")
    lines.append("")
    lines.append("next: python analysis/power.py --sigma-w %.4f --tasks-from configs/experiments/001-dev-loop.yaml"
                 % nf.sigma_w)
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_dir", help="directory of evaluator rollout JSONs")
    a = ap.parse_args(argv)
    print(report(estimate(load_rollouts(Path(a.json_dir)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
