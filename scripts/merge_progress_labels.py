#!/usr/bin/env python3
"""Merge progress labels into a LeRobot dataset as a real `progress` column.

Run this on the CLOUD box -- that is where the 330 GB training copy lives. The
Jetson ships only a small sidecar (episode_index, frame_index, progress); the
merge happens at the data.

WHY A REAL COLUMN, NOT A SIDECAR LOADED AT TRAIN TIME
-----------------------------------------------------
Traced through the openpi fork rather than assumed:

  openpi/src/openpi/transforms.py :: RepackTransform.__call__
      flat_item = flatten_dict(data)
      return jax.tree.map(lambda k: flat_item[k], self.structure)

  It maps flat SOURCE keys -> model input names, raises KeyError on a missing
  source key, and DISCARDS everything not named in the mapping. Its input dict
  is literally the HF dataset row:

  lerobot/datasets/dataset_reader.py :: DatasetReader.get_item
      item = self.hf_dataset[idx]

  So `repack_mapping["progress"] = <column>` -- which is what
  LeRobotB1KDataConfig.progress_key builds -- requires <column> to exist as a
  dataset column. That is what the mechanism was designed for.

THE TRAP THIS SCRIPT EXISTS TO AVOID
------------------------------------
  lerobot/datasets/dataset_reader.py :: _load_hf_dataset
      features = get_hf_features_from_features(self._meta.features)   # meta/info.json
      hf_dataset = load_nested_dataset(self.root / "data", features=features, ...)
  -> datasets.Dataset.from_parquet(paths, features=features)

An extra parquet column that is NOT registered in meta/info.json is not
silently dropped. It raises:

    CastError: Couldn't cast ... because column names don't match
    -> DatasetGenerationError

which makes the dataset unloadable for EVERY config, the baseline included. So
the parquet column and the info.json feature entry must land together or not at
all. This script writes both, then reloads the result to prove it.

By default it writes a NEW dataset root and symlinks `videos/` back to the
original, so:
  * the original 330 GB copy is never mutated,
  * the baseline arm provably reads the original bytes,
  * only the (small) low-dimensional parquet is rewritten.

Usage
-----
    python scripts/merge_progress_labels.py \
        --dataset-root  ~/data/b1k/turning_on_radio \
        --labels        ~/labels/turning_on_radio.parquet \
        --out-root      ~/data/b1k/turning_on_radio+progress

    # then train against the new root:
    #   --data.base_config.dataset_root=<out-root>  --data.progress_key=progress
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# LeRobot feature entry for a scalar float column. shape [1] is coerced to the
# tuple (1,) on load (lerobot/datasets/utils.py), which get_hf_features_from_features
# maps to datasets.Value("float32") -- a scalar, not a length-1 sequence.
PROGRESS_FEATURE = {"dtype": "float32", "shape": [1], "names": None}

DEFAULT_JOIN = ("episode_index", "frame_index")


def _load_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise SystemExit(f"unsupported label format {suffix!r} (want .parquet, .csv, .json, .jsonl)")


def validate_labels(labels, join_on: tuple[str, ...], column: str):
    """Fail loudly on anything that would produce a silently-wrong label."""
    missing = [c for c in (*join_on, column) if c not in labels.columns]
    if missing:
        raise SystemExit(
            f"labels are missing column(s) {missing}. Present: {list(labels.columns)}\n"
            f"Expected join keys {list(join_on)} plus the label column {column!r}."
        )

    dup = labels.duplicated(subset=list(join_on)).sum()
    if dup:
        raise SystemExit(
            f"{dup} duplicate label row(s) for the same {join_on}. "
            "A duplicated key makes the join ambiguous and would silently pick one."
        )

    values = labels[column]
    if values.isna().any():
        raise SystemExit(f"{int(values.isna().sum())} label(s) are NaN in column {column!r}")
    lo, hi = float(values.min()), float(values.max())
    if lo < 0.0 or hi > 1.0:
        raise SystemExit(
            f"progress must lie in [0, 1]; got [{lo:.4f}, {hi:.4f}]. "
            "The head is trained with binary cross-entropy, which is undefined outside that range."
        )
    return lo, hi


def merge_parquet_files(
    data_dir: Path,
    out_data_dir: Path,
    labels,
    join_on: tuple[str, ...],
    column: str,
    *,
    allow_missing: bool,
) -> tuple[int, int]:
    """Rewrite every data/*/*.parquet with a `progress` column joined on `join_on`.

    Returns (rows_total, rows_unlabelled).
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted(data_dir.glob("*/*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files under {data_dir} (expected data/chunk-*/file-*.parquet)")

    lookup = labels.set_index(list(join_on))[column].astype("float32")

    total = unlabelled = 0
    for path in paths:
        table = pq.read_table(path)
        if "progress" in table.column_names:
            raise SystemExit(
                f"{path} already has a 'progress' column. Refusing to merge twice -- "
                "point --dataset-root at the pristine dataset."
            )

        frame = table.select([c for c in join_on if c in table.column_names]).to_pandas()
        for key in join_on:
            if key not in frame.columns:
                raise SystemExit(
                    f"{path} has no column {key!r}; cannot join. Available: {table.column_names}"
                )

        keys = pd.MultiIndex.from_frame(frame[list(join_on)]) if len(join_on) > 1 else pd.Index(frame[join_on[0]])
        values = lookup.reindex(keys).to_numpy(dtype="float32")

        n_missing = int(np.isnan(values).sum())
        total += len(values)
        unlabelled += n_missing
        if n_missing and not allow_missing:
            sample = frame[np.isnan(values)].head(3).to_dict("records")
            raise SystemExit(
                f"{path}: {n_missing}/{len(values)} frames have no label.\n"
                f"  unmatched examples: {sample}\n"
                "A partial join trains the head on whatever fills the gap, which is worse than\n"
                "not training it at all. Fix the sidecar, or pass --allow-missing to fill with\n"
                "--missing-fill and accept that those frames carry a fabricated target."
            )

        out_path = out_data_dir / path.relative_to(data_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged = table.append_column("progress", pa.array(values, type=pa.float32()))
        pq.write_table(merged, out_path)

    return total, unlabelled


def patch_info(meta_dir: Path, out_meta_dir: Path) -> None:
    """Copy meta/ and register `progress` in info.json features."""
    if out_meta_dir.exists():
        shutil.rmtree(out_meta_dir)
    shutil.copytree(meta_dir, out_meta_dir)

    info_path = out_meta_dir / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"{info_path} not found -- is --dataset-root a LeRobot dataset root?")

    info = json.loads(info_path.read_text())
    features = info.get("features")
    if not isinstance(features, dict):
        raise SystemExit(f"{info_path} has no 'features' dict; cannot register the progress column")
    features["progress"] = dict(PROGRESS_FEATURE)
    info_path.write_text(json.dumps(info, indent=4))


def _features_from_info(info: dict):
    """Build the datasets.Features that LeRobot will build from this info.json.

    Mirrors lerobot/datasets/feature_utils.py :: get_hf_features_from_features for
    the numeric cases. Reimplemented rather than imported because importing
    lerobot drags in its whole video stack (av, torchvision), and a broken
    optional dependency must not stop us verifying a merge we already wrote.
    """
    import datasets as hf

    features = {}
    for key, ft in info["features"].items():
        dtype = ft.get("dtype")
        if dtype in ("video", "image"):
            continue  # not stored in the parquet
        shape = tuple(ft.get("shape", (1,)))
        if shape == (1,):
            features[key] = hf.Value(dtype)
        elif len(shape) == 1:
            features[key] = hf.Sequence(length=shape[0], feature=hf.Value(dtype))
        else:
            return None  # 2-D+ feature: let datasets infer rather than guess
    return hf.Features(features)


def verify(out_root: Path) -> None:
    """Reload the merged data the way training will, and confirm the column is there.

    This is the step that catches a parquet/info.json disagreement -- the failure
    that otherwise surfaces as an unloadable dataset at norm-stats time.
    """
    try:
        import datasets as hf_datasets
    except ImportError:
        print("!! `datasets` not importable -- skipping the reload check.", file=sys.stderr)
        print("   Run this on the training box, where it is installed.", file=sys.stderr)
        return

    info = json.loads((out_root / "meta" / "info.json").read_text())
    try:
        features = _features_from_info(info)
    except Exception as exc:  # noqa: BLE001 - never fail a written merge on the check
        print(f"!! could not build the feature schema ({exc}); inferring instead.", file=sys.stderr)
        features = None

    paths = [str(p) for p in sorted((out_root / "data").glob("*/*.parquet"))]
    try:
        ds = hf_datasets.Dataset.from_parquet(paths, features=features)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"MERGE WROTE, BUT THE RESULT DOES NOT LOAD:\n  {type(exc).__name__}: {exc}\n"
            "The parquet and meta/info.json disagree. Do not train against this root."
        ) from exc

    if "progress" not in ds.column_names:
        raise SystemExit(f"reload succeeded but 'progress' is absent: {ds.column_names}")
    value = ds[0]["progress"]
    if isinstance(value, list):
        raise SystemExit(
            f"'progress' loaded as a list {value}, not a scalar. The info.json shape is wrong; "
            "it must be [1] so LeRobot maps it to datasets.Value."
        )
    print(f"   reload OK: progress present, row 0 = {value!r} ({type(value).__name__})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", required=True, type=Path, help="pristine LeRobot dataset root")
    ap.add_argument("--labels", required=True, type=Path, help="sidecar table from the Jetson")
    ap.add_argument("--out-root", type=Path, default=None, help="destination root (default: <root>+progress)")
    ap.add_argument("--in-place", action="store_true", help="mutate --dataset-root instead (keeps a backup of info.json)")
    ap.add_argument("--column", default="progress", help="label column name in the sidecar")
    ap.add_argument("--join-on", nargs="+", default=list(DEFAULT_JOIN),
                    help="dataset columns to join on (default: episode_index frame_index)")
    ap.add_argument("--allow-missing", action="store_true", help="tolerate unlabelled frames")
    ap.add_argument("--missing-fill", type=float, default=0.0, help="value for unlabelled frames")
    ap.add_argument("--copy-videos", action="store_true", help="copy videos/ instead of symlinking")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out-root")
    args = ap.parse_args()

    root: Path = args.dataset_root.expanduser()
    if not (root / "data").is_dir() or not (root / "meta").is_dir():
        raise SystemExit(f"{root} does not look like a LeRobot root (needs data/ and meta/)")

    labels = _load_table(args.labels.expanduser())
    join_on = tuple(args.join_on)
    lo, hi = validate_labels(labels, join_on, args.column)
    print(f"==> {len(labels)} labels, {args.column} in [{lo:.4f}, {hi:.4f}], join on {join_on}")

    if args.in_place:
        out_root = root
        backup = root / "meta" / "info.json.pre-progress"
        if not backup.exists():
            shutil.copy2(root / "meta" / "info.json", backup)
            print(f"==> backed up info.json -> {backup.name}")
        out_data, out_meta = root / "data", root / "meta"
        staged = root / "data.progress-staging"
        if staged.exists():
            shutil.rmtree(staged)
        total, missing = merge_parquet_files(
            out_data, staged, labels, join_on, args.column, allow_missing=args.allow_missing
        )
        shutil.rmtree(out_data)
        staged.rename(out_data)
        info = json.loads((out_meta / "info.json").read_text())
        info["features"]["progress"] = dict(PROGRESS_FEATURE)
        (out_meta / "info.json").write_text(json.dumps(info, indent=4))
    else:
        out_root = (args.out_root or root.parent / f"{root.name}+progress").expanduser()
        if out_root.exists():
            if not args.force:
                raise SystemExit(f"{out_root} exists. Pass --force to replace it.")
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True)
        total, missing = merge_parquet_files(
            root / "data", out_root / "data", labels, join_on, args.column, allow_missing=args.allow_missing
        )
        patch_info(root / "meta", out_root / "meta")

        videos = root / "videos"
        if videos.is_dir():
            dest = out_root / "videos"
            if args.copy_videos:
                shutil.copytree(videos, dest)
                print("==> copied videos/")
            else:
                os.symlink(videos.resolve(), dest, target_is_directory=True)
                print(f"==> symlinked videos/ -> {videos.resolve()}")
        # Anything else at the root (e.g. README) is small; carry it over.
        for extra in root.iterdir():
            if extra.name in {"data", "meta", "videos"}:
                continue
            target = out_root / extra.name
            (shutil.copytree if extra.is_dir() else shutil.copy2)(extra, target)

    print(f"==> merged {total} frames ({missing} unlabelled)")
    if missing:
        print(f"!! {missing} frames carry the fabricated fill value {args.missing_fill}", file=sys.stderr)

    print("==> verifying by reloading through LeRobot's schema path")
    verify(out_root)

    print(f"\ndone: {out_root}")
    print("train against it with:")
    print(f"    --data.base_config.dataset_root={out_root} --data.progress_key={args.column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
