"""Build and validate the data roots for b1k training -- ONE place for all of it.

Three entry points set a dataset root on an openpi config: first_batch.py,
compute_norm_stats_b1k.py and train_b1k_rooted.py. Separate copies of that logic
are how one arm ends up on a different data path from the other, so the
validation lives here and all three call it.

THE ASYMMETRY (openpi data_loader.create_b1k_dataset)
    one repo_id -> dataset_root IS the dataset directory
    a list      -> dataset_root is the PARENT; each at <root>/<repo_id>, loaded by
                   MultiLeRobotDataset (the pi05_b1k config hard-sets
                   data_cls=LeRobotDataset, so multi-task must swap it)

THE SILENT FAILURE THIS REFUSES
    MultiLeRobotDataset disables every feature that is not present in ALL of its
    datasets, with only a warning. If one task lacks `progress`, arm B trains with
    the progress feature switched off for every task, and nothing raises.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class RootError(RuntimeError):
    pass


@dataclass(frozen=True)
class RootPlan:
    multi: bool
    dataset_root: Path
    repo_ids: list[str]
    dataset_dirs: list[Path]


def _info(d: Path) -> dict:
    p = d / "meta" / "info.json"
    if not p.is_file():
        raise RootError(f"{d} is not a LeRobot dataset (no meta/info.json)")
    return json.loads(p.read_text())


def validate_roots(dataset_root, repo_ids, *, progress_key: str | None = None,
                   protocol: bool = False) -> RootPlan:
    root = Path(dataset_root)
    ids = list(repo_ids)
    if not ids:
        raise RootError("no repo_ids given")
    if len(set(ids)) != len(ids):
        raise RootError(f"duplicate repo_ids {ids} -- a repeated task silently doubles its sampling weight")
    if not root.exists():
        raise RootError(f"dataset_root {root} does not exist")

    multi = len(ids) > 1
    dirs = [root / r for r in ids] if multi else [root]
    for r, d in zip(ids, dirs):
        if not d.is_dir():
            raise RootError(f"task {r!r}: expected a dataset at {d}, found nothing"
                            + (" (multi-task roots are the PARENT of per-task directories)" if multi else ""))

    infos = [_info(d) for d in dirs]

    if progress_key:
        missing = [r for r, i in zip(ids, infos) if progress_key not in i.get("features", {})]
        if missing:
            raise RootError(
                f"{progress_key!r} is not a registered feature of {missing}. MultiLeRobotDataset would "
                f"disable it for EVERY task and arm B would train with no progress target. "
                f"Merge labels into those roots first.")

    if protocol:
        for r, d, i in zip(ids, dirs, infos):
            man = d / "meta" / "progress_filter.json"
            if not man.is_file():
                raise RootError(f"task {r!r}: {d} has no meta/progress_filter.json -- a protocol run needs the "
                                f"merged, filtered, compacted root for EVERY task, in BOTH arms")
            eps = json.loads(man.read_text()).get("episodes_out")
            if eps != i.get("total_episodes"):
                raise RootError(f"task {r!r}: manifest episodes_out={eps} but info.json total_episodes="
                                f"{i.get('total_episodes')} -- dropped but not compacted, or a stale manifest")

    return RootPlan(multi=multi, dataset_root=root, repo_ids=ids, dataset_dirs=dirs)


def translate_dataset_kwargs(kwargs: dict, repo_ids: list[str], *, multi: bool) -> dict:
    """LeRobotDataset and MultiLeRobotDataset do not take the same kwargs.

    pi05_b1k sets `dataset_kwargs={"tolerance_s": 5e-4}` for the single-dataset
    class, and openpi's create_b1k_dataset forwards dataset_kwargs unchanged down
    BOTH paths -- so a list repo_id reaches MultiLeRobotDataset, whose parameter is
    `tolerances_s`, a dict keyed by repo_id. The two-task batch died on exactly
    that: "unexpected keyword argument 'tolerance_s'".
    """
    out = dict(kwargs)
    if multi and "tolerance_s" in out:
        tol = out.pop("tolerance_s")
        out.setdefault("tolerances_s", dict.fromkeys(repo_ids, tol))
    return out


def apply_to_config(cfg, plan: RootPlan, *, video_backend: str | None = None,
                    progress_key: str | None = None):
    """Return a copy of an openpi TrainConfig pointed at `plan`. Requires openpi."""
    import dataclasses

    base = cfg.data.base_config
    repo_id = plan.repo_ids if plan.multi else plan.repo_ids[0]
    changes = {"dataset_root": str(plan.dataset_root), "repo_id": repo_id}
    if plan.multi:
        from openpi.training import lerobot_compat
        changes["data_cls"] = lerobot_compat.MultiLeRobotDataset
    kw = translate_dataset_kwargs(dict(getattr(base, "dataset_kwargs", None) or {}),
                                  plan.repo_ids, multi=plan.multi)
    if video_backend:
        kw["video_backend"] = video_backend
    changes["dataset_kwargs"] = kw
    base = dataclasses.replace(base, **changes)
    data = dataclasses.replace(cfg.data, base_config=base, repo_id=repo_id)
    if progress_key:
        if hasattr(data, "progress_key"):
            data = dataclasses.replace(data, progress_key=progress_key)
        elif hasattr(base, "progress_key"):
            data = dataclasses.replace(data, base_config=dataclasses.replace(base, progress_key=progress_key))
        else:
            raise RootError("progress_key given but the config has no progress_key field (patch 0002 not applied?)")
    return dataclasses.replace(cfg, data=data)
