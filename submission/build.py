"""
Assemble and validate the challenge submission package.

Run this in Week 1 against a throwaway model, not in Week 7 against your best one.
The packaging pipeline is itself a thing that breaks -- Docker won't build, JSONs are
malformed, the wrapper gets rejected. Discover that early.

Required package contents (per the challenge submission guidelines):
  1. One metrics JSON per rollout, up to 1,000. Unmodified.
  2. The .py evaluation wrapper -- organizers inspect this manually to confirm the policy
     sees only RGB, depth and proprioception.
  3. The exact robot config (.yaml / .json) used for evaluation.
  4. A README with the precise evaluator command, wrapper path, robot config path, and
     Docker or IP-serving details.

Rollout MP4s are NOT zipped -- they are submitted as a link through the portal.

Usage
-----
    python -m submission.build --rollouts rollouts/final \\
        --wrapper policy/wrapper.py --robot-config configs/robot/r1pro.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

EXPECTED_ROLLOUTS = 1000  # 100 tasks x 10 instances x 1 rollout
LEADERBOARD_INSTANCES = set(range(10))  # 0-9 are the reported set


def validate_rollouts(json_dir: Path) -> tuple[list[Path], list[str]]:
    """Check the rollout JSONs before packaging. Returns (files, warnings)."""
    files = sorted(p for p in json_dir.glob("*.json") if p.name != "timing_manifest.json")
    warnings: list[str] = []

    if not files:
        raise SystemExit(f"no rollout JSONs in {json_dir}")

    if len(files) < EXPECTED_ROLLOUTS:
        warnings.append(
            f"{len(files)} rollouts, expected {EXPECTED_ROLLOUTS}. "
            "Partial submissions are allowed but missing instances score zero."
        )
    elif len(files) > EXPECTED_ROLLOUTS:
        warnings.append(f"{len(files)} rollouts exceeds the {EXPECTED_ROLLOUTS} cap.")

    seen: set[tuple] = set()
    off_leaderboard = 0
    for path in files:
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            warnings.append(f"{path.name}: malformed JSON ({exc})")
            continue

        for field in ("task", "instance_id", "q_score"):
            if field not in d:
                warnings.append(f"{path.name}: missing required field '{field}'")

        key = (d.get("task"), d.get("instance_id"), d.get("rollout_id"))
        if key in seen:
            warnings.append(f"duplicate rollout {key} -- do not submit repeated attempts")
        seen.add(key)

        if d.get("instance_id") not in LEADERBOARD_INSTANCES:
            off_leaderboard += 1

    if off_leaderboard:
        warnings.append(
            f"{off_leaderboard} rollout(s) use instances outside 0-9. "
            "Only 0-9 are reported to the leaderboard; 10-19 are your dev set."
        )

    return files, warnings


def write_readme(dest: Path, args, n_rollouts: int) -> None:
    dest.write_text(f"""# BEHAVIOR 2026 Challenge Submission

## Evaluator command

```
python -m omnigibson.eval.eval \\
  --task-name $TASK \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \\
  --num-rollouts 1 \\
  --env-wrapper {args.wrapper_target} \\
  --robot-config {Path(args.robot_config).name} \\
  --output-dir outputs/eval \\
  --write-video
```

## Contents

| File | Purpose |
|---|---|
| `json/` | {n_rollouts} rollout metrics files, unmodified |
| `{Path(args.wrapper).name}` | Evaluation wrapper. Exposes RGB, depth and proprioception only. |
| `{Path(args.robot_config).name}` | Exact robot config used for evaluation |

## Serving

**Mode:** {args.serving_mode}

{"The Docker image serves the policy over the websocket protocol. It runs within a single 24GB VRAM GPU." if args.serving_mode == "docker" else "IP-based serving. At least 50 ports are exposed for parallel evaluation."}

- Image / endpoint: {args.serving_detail or "TODO"}

## Rollout videos

{args.video_link or "TODO -- submit the link to all rollout MP4s through the portal."}

## Observation compliance

The wrapper exposes only RGB, depth and proprioception. No ground-truth segmentation,
object state, target object pose, full-scene point cloud, or robot global pose is read
from the simulator at evaluation time. The environment is not manipulated directly --
no teleporting the robot and no setting object states.
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the challenge submission package")
    ap.add_argument("--rollouts", required=True, type=Path,
                    help="rollout output dir (expects a json/ subdirectory)")
    ap.add_argument("--wrapper", required=True, type=Path, help="evaluation wrapper .py")
    ap.add_argument("--robot-config", required=True, type=Path, help="robot .yaml used")
    ap.add_argument("--wrapper-target", default="policy.wrapper.ChallengeWrapper",
                    help="import path passed to --env-wrapper")
    ap.add_argument("--serving-mode", choices=["docker", "ip"], default="docker")
    ap.add_argument("--serving-detail", default=None, help="image name or endpoint")
    ap.add_argument("--video-link", default=None, help="link to the rollout MP4s")
    ap.add_argument("--out", type=Path, default=Path("submission/dist/submission.zip"))
    ap.add_argument("--force", action="store_true", help="package despite warnings")
    args = ap.parse_args()

    json_dir = args.rollouts / "json"
    if not json_dir.is_dir():
        json_dir = args.rollouts

    files, warnings = validate_rollouts(json_dir)

    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  ! {w}")
        blocking = [w for w in warnings if "malformed" in w or "missing required" in w]
        if blocking and not args.force:
            print("\nblocking problems found. fix them, or pass --force.")
            return 1
        print()

    for path in (args.wrapper, args.robot_config):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")

    staging = args.out.parent / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "json").mkdir(parents=True)

    for path in files:
        shutil.copy2(path, staging / "json" / path.name)
    shutil.copy2(args.wrapper, staging / args.wrapper.name)
    shutil.copy2(args.robot_config, staging / args.robot_config.name)
    write_readme(staging / "README.md", args, len(files))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]
    size_mb = args.out.stat().st_size / 1e6

    print(f"packaged {len(files)} rollouts -> {args.out}")
    print(f"  size   {size_mb:.1f} MB")
    print(f"  sha256 {digest}")
    print("\nreminder: submit the rollout MP4 link separately through the portal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
