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
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# VERIFIED against BEHAVIOR-1K v3.9.2 (omnigibson/eval/utils/eval_utils.py and
# utils/score_utils.py). These replace the earlier 0-9 / 1,000-rollout figures,
# which the evaluator source contradicts:
#
#   TEST_INSTANCE_IDS = list(range(301, 341))     # 40 test instances
#   NUM_PUBLIC_TEST_INSTANCES = 20                # public split = 301..320
#
# `--instance-indices` are indices INTO the split; the rollout JSON and its
# filename carry the resolved id (301..320 for public_test).
#
# compute_final_q_score averages with a FIXED denominator:
#     q_score_avg[task] = sum(q_score[task].values()) / n_instances_per_task
# with n_instances_per_task = 20 for the public testset. Covering only half the
# instances therefore halves the reported score -- the missing ones are zeros.
TEST_INSTANCE_IDS = list(range(301, 341))
NUM_PUBLIC_TEST_INSTANCES = 20
PUBLIC_INSTANCES = set(TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES])
HIDDEN_INSTANCES = set(TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:])
N_TASKS = 100
EXPECTED_ROLLOUTS = N_TASKS * NUM_PUBLIC_TEST_INSTANCES  # 100 x 20 x 1 rollout

# The official scorer parses the submission directory name as exactly five
# dot-separated fields (utils/score_utils.py l.249-254):
#     <track>.<testset>.<team>.<affiliation>.<date>/json/*.json
VALID_TRACKS = ("standard", "privileged")
VALID_TESTSETS = ("public", "hidden")
FILENAME_RE = re.compile(r"^(?P<task>.+)_(?P<instance>\d+)_(?P<rollout>\d+)\.json$")


def validate_rollouts(
    json_dir: Path, testset: str, intended_tasks: list[str] | None = None
) -> tuple[list[Path], list[str]]:
    """Check the rollout JSONs before packaging. Returns (files, warnings).

    The checks mirror what the official scorer asserts, so a problem shows up
    here rather than as an AssertionError in someone else's pipeline.

    `intended_tasks` switches on PARTIAL mode. Partial submissions are legal and
    on a small budget they are the plan, not an accident: we submit only the
    tasks we trained. Without this the count check fires on every build ("40
    rollouts, expected 2000") and a warning that always fires is a warning
    nobody reads -- which is how a genuine shortfall gets missed.

    In partial mode the expected count is computed from the intended list, and
    a real shortfall -- a task that produced fewer rollouts than its instances,
    or produced none at all -- still warns. Tasks outside the list also warn,
    because that means the run did something the plan did not ask for.
    """
    all_json = sorted(json_dir.glob("*.json"))
    files = [p for p in all_json if p.name != "timing_manifest.json"]
    warnings: list[str] = []

    if not files:
        raise SystemExit(f"no rollout JSONs in {json_dir}")

    stray = [p.name for p in all_json if p not in files]
    if stray:
        warnings.append(
            f"excluded {stray} from the package. compute_final_q_score asserts EVERY "
            "*.json in json/ is a valid rollout filename, so a stray file makes it raise."
        )

    expected_instances = PUBLIC_INSTANCES if testset == "public" else HIDDEN_INSTANCES
    n_per_task = len(expected_instances)
    partial = intended_tasks is not None
    expected_total = len(intended_tasks) * n_per_task if partial else EXPECTED_ROLLOUTS

    if len(files) > expected_total:
        warnings.append(
            f"{len(files)} rollouts exceeds the {expected_total} expected"
            + (" for the intended task list." if partial else f" ({EXPECTED_ROLLOUTS} cap).")
        )
    elif len(files) < expected_total:
        if partial:
            warnings.append(
                f"{len(files)} rollouts, expected {expected_total} "
                f"({len(intended_tasks)} intended tasks x {n_per_task} instances). "
                "This is a shortfall WITHIN the intended set, not the intentional "
                "partial -- some rollouts did not run."
            )
        else:
            warnings.append(
                f"{len(files)} rollouts, expected {EXPECTED_ROLLOUTS} "
                f"({N_TASKS} tasks x {n_per_task} instances). Missing rollouts are NOT skipped: "
                f"compute_final_q_score divides by {n_per_task} per task regardless, so each "
                "missing instance is scored as a zero. If this is a deliberate partial "
                "submission, pass --intended-task (repeatable) or --intended-tasks-from."
            )

    seen: set[tuple] = set()
    bad_names: list[str] = []
    off_split: list[int] = []
    tasks: set[str] = set()
    per_task: dict[str, int] = {}

    for path in files:
        match = FILENAME_RE.match(path.name)
        if not match:
            bad_names.append(path.name)
        else:
            tasks.add(match["task"])
            per_task[match["task"]] = per_task.get(match["task"], 0) + 1
            instance = int(match["instance"])
            if instance not in expected_instances:
                off_split.append(instance)

        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            warnings.append(f"{path.name}: malformed JSON ({exc})")
            continue

        for field in ("task", "instance_id", "q_score"):
            if field not in d:
                warnings.append(f"{path.name}: missing required field '{field}'")

        # The scorer reads q_score.final and time.normalized_time by path.
        if "final" not in (d.get("q_score") or {}):
            warnings.append(f"{path.name}: q_score has no 'final' key")
        if "normalized_time" not in (d.get("time") or {}):
            warnings.append(f"{path.name}: time has no 'normalized_time' key")

        # A filename/content mismatch scores the rollout under the wrong instance.
        if match and d.get("instance_id") is not None and int(d["instance_id"]) != int(match["instance"]):
            warnings.append(
                f"{path.name}: filename says instance {match['instance']} but the JSON says "
                f"{d['instance_id']}. The scorer keys off the FILENAME."
            )

        key = (d.get("task"), d.get("instance_id"), d.get("rollout_id"))
        if key in seen:
            warnings.append(f"duplicate rollout {key} -- do not submit repeated attempts")
        seen.add(key)

    if bad_names:
        warnings.append(
            f"{len(bad_names)} file(s) do not match <task>_<instance>_<rollout>.json, "
            f"e.g. {bad_names[:3]}. compute_final_q_score asserts on this and will raise."
        )
    if off_split:
        lo, hi = min(expected_instances), max(expected_instances)
        warnings.append(
            f"{len(off_split)} rollout(s) use instance ids outside the {testset} split "
            f"({lo}-{hi}), e.g. {sorted(set(off_split))[:5]}. Note --instance-indices are "
            "INDICES into the split; the evaluator writes the resolved id."
        )
    if partial:
        intended = set(intended_tasks)
        missing = sorted(intended - tasks)
        unexpected = sorted(tasks - intended)
        if missing:
            warnings.append(
                f"intended task(s) with NO rollouts at all: {missing}. "
                "These score zero."
            )
        if unexpected:
            warnings.append(
                f"rollouts for task(s) not in the intended list: {unexpected}. "
                "The run did something the plan did not ask for."
            )
        for task in sorted(intended & tasks):
            got = per_task.get(task, 0)
            if got < n_per_task:
                warnings.append(
                    f"{task}: {got}/{n_per_task} instances. Each missing instance is a zero."
                )
        covered = len(intended & tasks)
        warnings.append(
            f"PARTIAL submission: {covered}/{N_TASKS} tasks covered. Q is averaged over all "
            f"{N_TASKS} tasks, so the other {N_TASKS - covered} score zero and this run's "
            f"ceiling is {covered / N_TASKS:.3f} even at Q=1.0 on every task submitted."
        )
    elif len(tasks) < N_TASKS:
        warnings.append(
            f"{len(tasks)} distinct task(s) present, expected {N_TASKS}. "
            "Absent tasks score zero and are averaged in."
        )

    return files, warnings


def write_readme(dest: Path, args, n_rollouts: int, extras: list[Path], submission_name: str) -> None:
    """The organizers require a README explaining how to evaluate the policy.

    Everything quoted here is taken from BEHAVIOR-1K v3.9.2 rather than from
    memory: the evaluator's real CLI (omnigibson/eval/eval.py), its default
    wrapper, and the instance-split semantics.
    """
    extra_rows = "\n".join(
        f"| `artifacts/{e.name}` | Precomputed artifact, shipped inside the package |" for e in extras
    )
    serving = (
        "The Docker image serves the policy over the websocket protocol described below, "
        "within a single 24GB VRAM GPU."
        if args.serving_mode == "docker"
        else "IP-based serving. At least 50 ports are exposed so instances can be evaluated in parallel."
    )
    dest.write_text(f"""# BEHAVIOR 2026 Challenge Submission

Package: `{submission_name}`
Track: **{args.track}** · Test set: **{args.testset}** · Team: **{args.team}** ({args.affiliation})

## What this is

A π₀.₅-based policy served over the websocket protocol the BEHAVIOR-1K evaluator
speaks. `json/` holds {n_rollouts} unmodified per-rollout metrics files.

## How to evaluate this policy

### 1. Start the policy server

{serving}

- Image / endpoint: {args.serving_detail or "TODO -- fill in before submitting"}

The server must implement the protocol in
`omnigibson/eval/utils/network_utils.py`:

- answer `GET /healthz` with 2xx on the same host:port;
- on websocket connect, send **one msgpack metadata frame first** — the client
  blocks on `unpackb(conn.recv())` before sending anything;
- for each observation frame, reply `{{"action": <ndarray>}}`. The value must
  decode to a real `numpy.ndarray` (the client calls `th.from_numpy` on it), and
  its width must equal `robot.action_dim` (23 for R1Pro);
- ndarrays use the evaluator's own msgpack extension
  (`{{b"__ndarray__": True, b"data", b"dtype", b"shape"}}`), **not** msgpack-numpy;
- a `{{"reset": True}}` frame expects **no reply**.

### 2. Run the evaluator

```
python -m omnigibson.eval.eval \\
  --task-name $TASK \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --mode {args.testset}_test \\
  --instance-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \\
  --num-rollouts 1 \\
  --env-wrapper {args.wrapper_target} \\
  --robot-config {Path(args.robot_config).name} \\
  --output-dir outputs/eval \\
  --write-video
```

`--instance-indices` are **indices into the split**, not instance ids. For
`--mode {args.testset}_test` they index `TEST_INSTANCE_IDS[{"" if args.testset == "public" else "20"}:{"20" if args.testset == "public" else ""}]`,
i.e. instances **{min(PUBLIC_INSTANCES) if args.testset == "public" else min(HIDDEN_INSTANCES)}–{max(PUBLIC_INSTANCES) if args.testset == "public" else max(HIDDEN_INSTANCES)}**. Result filenames carry the resolved id.

### 3. Score

```
python -m omnigibson.eval.utils.score_utils --input-dir {submission_name} --output-dir scores/
```

The directory name is parsed as `<track>.<testset>.<team>.<affiliation>.<date>`,
which is why the package is laid out that way.

## Contents

| File | Purpose |
|---|---|
| `json/` | {n_rollouts} rollout metrics files, unmodified |
| `{Path(args.wrapper).name}` | Evaluation wrapper. Exposes RGB, depth and proprioception only. |
| `{Path(args.robot_config).name}` | Exact robot config used for evaluation |
{extra_rows}

## Rollout videos

{args.video_link or "TODO -- submit the link to all rollout MP4s through the portal."}

MP4s are not zipped; they are submitted as a link through the portal.

## Observation compliance

The wrapper exposes only RGB, depth and proprioception. No ground-truth
segmentation, object state, target object pose, full-scene point cloud, or robot
global pose is read from the simulator at evaluation time. The environment is not
manipulated directly — no teleporting the robot and no setting object states.
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the challenge submission package")
    ap.add_argument("--rollouts", required=True, type=Path,
                    help="rollout output dir (expects a json/ subdirectory)")
    ap.add_argument("--wrapper", required=True, type=Path, help="evaluation wrapper .py")
    ap.add_argument("--robot-config", required=True, type=Path, help="robot .yaml used")
    ap.add_argument("--wrapper-target", default="policy.wrapper.ChallengeWrapper",
                    help="import path passed to --env-wrapper")
    # The scorer parses the directory name as <track>.<testset>.<team>.<affiliation>.<date>.
    ap.add_argument("--track", choices=VALID_TRACKS, default="standard")
    ap.add_argument("--testset", choices=VALID_TESTSETS, default="public")
    ap.add_argument("--team", required=True, help="team name (no dots)")
    ap.add_argument("--affiliation", required=True, help="affiliation (no dots)")
    ap.add_argument("--date", default=None, help="YYYYMMDD; defaults to today (UTC)")
    ap.add_argument("--intended-task", action="append", default=[], metavar="TASK",
                    help="declare a deliberate PARTIAL submission: the task(s) we meant to "
                         "submit. Repeatable. Validation then runs against this list, so a "
                         "real shortfall still warns but the intentional partial does not.")
    ap.add_argument("--intended-tasks-from", type=Path, default=None, metavar="YAML",
                    help="read the intended task list from an experiment config's `tasks:` "
                         "key, e.g. configs/experiments/001-dev-loop.yaml")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="precomputed artifact to ship inside the package (repeatable), "
                         "e.g. norm_stats.json")
    ap.add_argument("--serving-mode", choices=["docker", "ip"], default="docker")
    ap.add_argument("--serving-detail", default=None, help="image name or endpoint")
    ap.add_argument("--video-link", default=None, help="link to the rollout MP4s")
    ap.add_argument("--out", type=Path, default=None,
                    help="zip path (default: submission/dist/<package name>.zip)")
    ap.add_argument("--force", action="store_true", help="package despite warnings")
    args = ap.parse_args()

    date = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    for field, value in (("team", args.team), ("affiliation", args.affiliation), ("date", date)):
        if not value or "." in value:
            raise SystemExit(
                f"--{field} must be non-empty and contain no '.': the scorer splits the "
                f"directory name on '.' into exactly 5 fields, got {value!r}"
            )
    submission_name = f"{args.track}.{args.testset}.{args.team}.{args.affiliation}.{date}"
    out = args.out or Path("submission/dist") / f"{submission_name}.zip"

    json_dir = args.rollouts / "json"
    if not json_dir.is_dir():
        json_dir = args.rollouts

    intended = list(args.intended_task)
    if args.intended_tasks_from:
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            raise SystemExit("--intended-tasks-from needs PyYAML (pip install -r requirements-tools.txt)")
        cfg = yaml.safe_load(args.intended_tasks_from.read_text()) or {}
        from_cfg = cfg.get("tasks") or []
        if not from_cfg:
            raise SystemExit(f"{args.intended_tasks_from} has no non-empty `tasks:` list")
        intended += [t for t in from_cfg if t not in intended]
    intended_tasks = intended or None

    files, warnings = validate_rollouts(json_dir, args.testset, intended_tasks)

    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  ! {w}")
        blocking = [
            w for w in warnings
            if "malformed" in w or "missing required" in w or "will raise" in w or "scorer keys off" in w
        ]
        if blocking and not args.force:
            print("\nblocking problems found. fix them, or pass --force.")
            return 1
        print()

    for path in (args.wrapper, args.robot_config, *args.extra):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")

    staging = out.parent / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    # The scorer expects <name>/json/*.json, so the package root is that directory.
    root = staging / submission_name
    (root / "json").mkdir(parents=True)

    for path in files:
        shutil.copy2(path, root / "json" / path.name)
    shutil.copy2(args.wrapper, root / args.wrapper.name)
    shutil.copy2(args.robot_config, root / args.robot_config.name)

    if args.extra:
        (root / "artifacts").mkdir()
        for path in args.extra:
            shutil.copy2(path, root / "artifacts" / path.name)

    write_readme(root / "README.md", args, len(files), list(args.extra), submission_name)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    size_mb = out.stat().st_size / 1e6

    if intended_tasks:
        print(f"PARTIAL submission: {len(intended_tasks)} intended task(s) "
              f"-> ceiling {len(intended_tasks) / N_TASKS:.3f}")
    print(f"packaged {len(files)} rollouts -> {out}")
    print(f"  package  {submission_name}/")
    print(f"  size     {size_mb:.1f} MB")
    print(f"  sha256   {digest}")
    if args.extra:
        print(f"  artifacts {', '.join(p.name for p in args.extra)}")
    print("\nreminder: submit the rollout MP4 link separately through the portal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
