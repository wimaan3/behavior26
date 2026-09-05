"""
Parallel rollout launcher for the BEHAVIOR 2026 challenge.

Why this exists
---------------
A full public submission is 100 tasks x 20 instances x 1 rollout = 2,000 rollouts.
At the organizers' published throughput (~13.5 FPS for full-res RGB+depth) plus 150-300s
scene load per trial, one rollout runs roughly 20-25 minutes. That puts a single full
evaluation at ~350-420 GPU-hours -- over two weeks on one card.

So evaluation has to be fanned out across workers, and it has to be resumable.
This module does that by shelling out to the official evaluator, one subprocess per job.

It also records wall-clock per job, which is the measurement everything else depends on
(see docs/WEEK1_CHARTER.md). Do not remove the timing manifest.

Port assignment
---------------
Each concurrent eval worker needs its own policy server connection. The challenge's
IP-submission mode requires >=50 exposed ports for exactly this reason. Worker i talks to
``base_port + i``, so you must have that many policy servers running (or one server bound
across that range) before launching.

Usage
-----
    python -m harness.launch --config configs/experiments/000-baseline-smoke.yaml
    python -m harness.launch --config <cfg> --workers 8 --base-port 8000
    python -m harness.launch --config <cfg> --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


# The official entry point. Verify against your BEHAVIOR-1K checkout before a real run:
#   python -m omnigibson.eval.eval --help
#
# Overridable so the whole harness can be exercised without a GPU against
# tests/mock_evaluator.py, which speaks the same CLI. Precedence:
#   --eval-module  >  config key `eval_module`  >  $BEHAVIOR_EVAL_MODULE  >  the real one
EVAL_MODULE = os.environ.get("BEHAVIOR_EVAL_MODULE", "omnigibson.eval.eval")


# VERIFIED against BEHAVIOR-1K v3.9.2 (omnigibson/eval/utils/eval_utils.py,
# omnigibson/eval/evaluator.py :: resolve_instance_ids).
#
# `--instance-indices` are INDICES INTO A SPLIT, not instance ids. For
# public_test, index 0 is instance 301. The evaluator writes the RESOLVED id into
# both the rollout JSON and its filename, so anything that matches results back
# to jobs -- i.e. --resume -- has to resolve too. Before this existed, resume
# compared indices against resolved ids, matched nothing, and silently re-ran
# every job on a sweep that was already half done.
TEST_INSTANCE_IDS = list(range(301, 341))
NUM_PUBLIC_TEST_INSTANCES = 20
EVAL_MODES = ("train", "public_test", "hidden_test")


def resolve_instance_ids(instance_indices: list[int], mode: str) -> list[int]:
    """Mirror of omnigibson.eval.evaluator.resolve_instance_ids."""
    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")
    if mode == "train":
        return [int(i) for i in instance_indices]
    split = (
        TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES]
        if mode == "public_test"
        else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:]
    )
    bad = [i for i in instance_indices if not 0 <= i < len(split)]
    if bad:
        raise ValueError(
            f"instance indices {bad} out of range for mode {mode!r}: must be in "
            f"range({len(split)}). These index the split; the ids are "
            f"{split[0]}-{split[-1]}."
        )
    return [int(split[i]) for i in instance_indices]


@dataclass
class Job:
    """One evaluator invocation: a single task over one or more instances."""

    task: str
    instances: list[int]           # indices, as passed to --instance-indices
    resolved: list[int]            # the ids the evaluator will actually write
    worker_id: int = -1

    @property
    def key(self) -> str:
        lo, hi = min(self.instances), max(self.instances)
        return f"{self.task}__i{lo}-{hi}"


@dataclass
class JobResult:
    key: str
    task: str
    instances: list[int]
    returncode: int
    wall_clock_s: float
    started_at: float
    stdout_tail: str = ""


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for required in ("name", "tasks", "instances"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required key '{required}'")
    mode = cfg.get("mode", "public_test")
    if mode not in EVAL_MODES:
        raise ValueError(f"{path}: mode must be one of {EVAL_MODES}, got {mode!r}")
    return cfg


def build_jobs(cfg: dict, instances_per_job: int) -> list[Job]:
    """Split the (task x instance) grid into evaluator invocations.

    Grouping instances into one subprocess amortizes process startup and the OmniGibson
    import. Scene load is per-trial regardless, so grouping costs little.
    """
    tasks: list[str] = cfg["tasks"]
    instances: list[int] = cfg["instances"]
    mode: str = cfg.get("mode", "public_test")

    def make(task: str, chunk: list[int]) -> Job:
        return Job(task=task, instances=list(chunk), resolved=resolve_instance_ids(chunk, mode))

    jobs: list[Job] = []
    for task in tasks:
        if instances_per_job <= 0:
            jobs.append(make(task, list(instances)))
        else:
            for i in range(0, len(instances), instances_per_job):
                jobs.append(make(task, instances[i : i + instances_per_job]))
    return jobs


def completed_keys(output_dir: Path) -> set[tuple[str, int]]:
    """Index the (task, instance_id) pairs that already have a result on disk.

    Read from the JSON *contents*, not the filenames. The evaluator's filename convention
    is not documented, and matching on it by substring is actively dangerous: "1" is a
    substring of "inst10" and "0" of "rollout0", so a filename-based check reports
    instances as complete that were never run. On a resumed 2,000-rollout sweep those
    silently become missing instances, and missing instances score ZERO.

    Built once per launch rather than per job -- this is O(files), not O(jobs x files).
    """
    json_dir = output_dir / "json"
    if not json_dir.is_dir():
        return set()

    keys: set[tuple[str, int]] = set()
    for path in json_dir.glob("*.json"):
        if path.name == "timing_manifest.json":
            continue
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # A truncated JSON means the rollout was killed mid-write. Do NOT treat it as
            # complete -- leaving it out of the index makes the job re-run, which is the
            # entire point of resume.
            print(f"  ! ignoring unreadable {path.name} ({exc}); its job will re-run")
            continue
        task, inst = d.get("task"), d.get("instance_id")
        if task is None or inst is None:
            print(f"  ! {path.name} has no task/instance_id; its job will re-run")
            continue
        try:
            keys.add((str(task), int(inst)))
        except (TypeError, ValueError):
            print(f"  ! {path.name} has a non-integer instance_id {inst!r}; its job will re-run")
    return keys


def already_done(job: Job, done: set[tuple[str, int]]) -> bool:
    """Resume support: a job is complete only when EVERY instance it covers has a result.

    Partially-completed jobs re-run in full. That re-does some finished rollouts, which is
    the cheap mistake; the expensive one is skipping a rollout that never happened.
    """
    # job.resolved, NOT job.instances: the evaluator writes resolved ids.
    return all((job.task, inst) in done for inst in job.resolved)


def build_command(job: Job, cfg: dict, port: int, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", cfg.get("eval_module") or EVAL_MODULE,
        "--task-name", job.task,
        "--host", cfg.get("host", "127.0.0.1"),
        "--port", str(port),
        "--instance-indices", *[str(i) for i in job.instances],
        "--num-rollouts", str(cfg.get("num_rollouts", 1)),
        "--output-dir", str(output_dir),
        # Defaults to public_test in the evaluator too, but pass it explicitly:
        # a dev loop MUST use train, since every public_test instance is scored.
        "--mode", cfg.get("mode", "public_test"),
    ]

    wrapper = cfg.get("env_wrapper")
    if wrapper:
        cmd += ["--env-wrapper", wrapper]

    robot_config = cfg.get("robot_config")
    if robot_config:
        cmd += ["--robot-config", str(robot_config)]

    # Omit max_steps to get the task-specific default (1.5x mean human demo length).
    # Only set it deliberately -- shortening it changes what your Q means.
    if cfg.get("max_steps"):
        cmd += ["--max-steps", str(cfg["max_steps"])]

    if cfg.get("write_video", True):
        cmd.append("--write-video")
    if cfg.get("video_fps"):
        cmd += ["--video-fps", str(cfg["video_fps"])]

    cmd.append("--headless" if cfg.get("headless", True) else "--no-headless")

    # Pass-through for evaluator flags the config does not model (and for the mock's
    # own knobs, e.g. --fast --target-q 0.2).
    cmd += [str(a) for a in cfg.get("extra_eval_args", [])]
    return cmd


def run_job(job: Job, cfg: dict, port: int, gpu_id: int | None, output_dir: Path,
            log_dir: Path, dry_run: bool) -> JobResult:
    cmd = build_command(job, cfg, port, output_dir)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # OmniGibson picks its render device separately; without this it can land on the
        # wrong GPU and hang at "HydraEngine rtx failed creating scene renderer".
        env["OMNIGIBSON_GPU_ID"] = "0"

    if dry_run:
        print(f"[dry-run] w{job.worker_id} :: {shlex.join(cmd)}")
        return JobResult(job.key, job.task, job.instances, 0, 0.0, time.time())

    log_path = log_dir / f"{job.key}.log"
    started = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    tail = ""
    try:
        tail = "".join(open(log_path).readlines()[-15:])
    except OSError:
        pass

    status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    per_rollout = elapsed / max(len(job.instances), 1)
    print(f"[w{job.worker_id}] {job.key}: {status} "
          f"{elapsed/60:.1f} min total, {per_rollout/60:.1f} min/rollout")

    return JobResult(job.key, job.task, job.instances, proc.returncode, elapsed, started, tail)


def worker_loop(worker_id: int, job_q: "queue.Queue[Job]", results: list[JobResult],
                lock: threading.Lock, cfg: dict, base_port: int, gpus: list[int] | None,
                output_dir: Path, log_dir: Path, dry_run: bool) -> None:
    port = base_port + worker_id
    gpu_id = gpus[worker_id % len(gpus)] if gpus else None

    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        job.worker_id = worker_id
        try:
            result = run_job(job, cfg, port, gpu_id, output_dir, log_dir, dry_run)
        except Exception as exc:  # keep one bad job from killing the sweep
            print(f"[w{worker_id}] {job.key}: EXCEPTION {exc}", file=sys.stderr)
            result = JobResult(job.key, job.task, job.instances, -1, 0.0, time.time(), str(exc))
        with lock:
            results.append(result)
        job_q.task_done()


def write_timing_manifest(results: list[JobResult], path: Path, total_wall: float) -> None:
    """THE NUMBER. Everything downstream is planned off this file."""
    completed = [r for r in results if r.returncode == 0 and r.wall_clock_s > 0]
    rollouts = sum(len(r.instances) for r in completed)
    total_compute = sum(r.wall_clock_s for r in completed)
    per_rollout = (total_compute / rollouts) if rollouts else 0.0

    manifest = {
        "rollouts_completed": rollouts,
        "jobs_completed": len(completed),
        "jobs_failed": len(results) - len(completed),
        "wall_clock_total_s": round(total_wall, 1),
        "compute_time_total_s": round(total_compute, 1),
        "mean_seconds_per_rollout": round(per_rollout, 1),
        "mean_minutes_per_rollout": round(per_rollout / 60, 2),
        "projected_gpu_hours_for_1000_rollouts": round(per_rollout * 1000 / 3600, 1),
        "jobs": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 62)
    print(f"  rollouts completed : {rollouts}")
    print(f"  failed jobs        : {manifest['jobs_failed']}")
    print(f"  mean per rollout   : {manifest['mean_minutes_per_rollout']} min")
    print(f"  -> 1,000 rollouts  : {manifest['projected_gpu_hours_for_1000_rollouts']} GPU-hours")
    print("=" * 62)
    print(f"  manifest: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel BEHAVIOR rollout launcher")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent evaluator processes; each needs its own policy server port")
    ap.add_argument("--base-port", type=int, default=8000,
                    help="worker i connects to base_port + i")
    ap.add_argument("--gpus", type=str, default=None,
                    help="comma-separated GPU ids to round-robin across, e.g. 0,1,2,3")
    ap.add_argument("--instances-per-job", type=int, default=0,
                    help="instances per evaluator process; 0 = all instances of a task in one")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: rollouts/<experiment name>")
    ap.add_argument("--eval-module", default=None,
                    help="evaluator module to run; defaults to the official one. "
                         "Set tests.mock_evaluator to exercise the harness without a GPU")
    ap.add_argument("--extra-eval-arg", action="append", default=[], metavar="ARG",
                    help="extra argument appended to every evaluator command; repeatable")
    ap.add_argument("--resume", action="store_true", help="skip jobs whose JSONs already exist")
    ap.add_argument("--dry-run", action="store_true", help="print commands and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.eval_module:
        cfg["eval_module"] = args.eval_module
    if args.extra_eval_arg:
        cfg["extra_eval_args"] = list(args.extra_eval_arg)
    output_dir = args.output_dir or Path("rollouts") / cfg["name"]
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    jobs = build_jobs(cfg, args.instances_per_job)

    if args.resume:
        before = len(jobs)
        done = completed_keys(output_dir)
        jobs = [j for j in jobs if not already_done(j, done)]
        print(f"resume: {len(done)} rollout(s) already on disk, "
              f"skipping {before - len(jobs)} of {before} job(s)")

    if not jobs:
        print("nothing to do.")
        return 0

    n_rollouts = sum(len(j.instances) for j in jobs)
    print(f"experiment : {cfg['name']}")
    print(f"jobs       : {len(jobs)}  ({n_rollouts} rollouts)")
    print(f"workers    : {args.workers}  ports {args.base_port}-{args.base_port + args.workers - 1}")
    print(f"output     : {output_dir}\n")

    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results: list[JobResult] = []
    lock = threading.Lock()
    started = time.time()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(i, job_q, results, lock, cfg, args.base_port, gpus,
                  output_dir, log_dir, args.dry_run),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.dry_run:
        write_timing_manifest(results, output_dir / "timing_manifest.json", time.time() - started)

    return 1 if any(r.returncode != 0 for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
