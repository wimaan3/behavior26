# Slicing the combined corpus into per-task datasets

## The gate: pi0.5's b1k config IS multi-task. Confirmed in code, 2026-09-14.

The k argument — that 8 tasks costs the same gradient steps as 2, so k is bought
with data rather than compute — rests on this. It holds:

- **`repo_id` is `str | List[str]`** (`training/config.py:70,181`).
- **A list switches to the multi-dataset path.** `data_loader.py::create_b1k_dataset`:
  a list passes `repo_ids=` (plural) to `data_cls`; a string passes `repo_id=`.
  Each repo resolves at `<dataset_root>/<repo_id>`.
- **Prompts are per-sample, not per-config.** The b1k config sets
  `prompt_from_task=True`, which wraps the dataset in
  `PromptFromLeRobotTask(tasks_from_metadata(...))`; `B1KInputs` then forwards
  `data["prompt"]` (`b1k_policy.py:99`).
- **`LeRobotB1KDataConfig.create()` is task-agnostic.** Every transform is built
  from the ROBOT config. Nothing in the pipeline is per-task except which
  dataset(s) get loaded.

`turning_on_radio` is simply the fork's demo default, not a structural limit.

## The trap that decides the converter's design

`PromptFromLeRobotTask` resolves prompts by integer lookup:

```python
task_index = int(data["task_index"])
if (prompt := self.tasks.get(task_index)) is None:
    raise ValueError(...)
```

and when `repo_id` is a list, `create_b1k_dataset` builds that mapping from
**`repo_id[0]`'s metadata alone**:

```python
dataset_meta = LeRobotDatasetMetadata(repo_id=data_config.repo_id[0], ...)
...
dataset = TransformedDataset(dataset, [PromptFromLeRobotTask(tasks_from_metadata(dataset_meta))])
```

So the obvious converter — renumber each slice's `task_index` from 0 — is
**silently catastrophic**: every slice emits `task_index=0`, every sample in an
8-task run receives the FIRST task's prompt, and nothing errors. The model trains
happily on systematically mislabelled instructions.

**Therefore: preserve the GLOBAL `task_index`, and keep the FULL 100-task
`tasks.parquet` in every slice.** Then index→name is identical across slices,
`repo_id[0]`'s table covers every task, and a genuine mismatch raises loudly
instead of resolving to the wrong prompt.

This is the "index renumbering goes wrong silently" case, and the fix is to not
renumber at all.

## Layout: v3.0, and both sides agree

Not a design choice — read from the artifacts:

| | |
|---|---|
| dataset `meta/info.json` | `"codebase_version": "v3.0"` |
| openpi `pyproject.toml` | `lerobot = { git = "https://github.com/wensi-ai/lerobot", branch = "release/b1k" }` |
| OmniGibson `setup.py:58` | `lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k` |

**Identical pin on both sides**, so there is no eval-vs-training LeRobot split to
reconcile. Target v3.0.

v3 stores many episodes per parquet file (`data/chunk-NNN/file-NNN.parquet`) with
episode metadata in `meta/episodes/chunk-NNN/`, rather than v2's one file per
episode. Chunk index == task index in this corpus (`download_data.sh`: "Tasks are
chunks"), verified against `meta/tasks.parquet`.

## Norm stats

Recomputed over the slice with `scripts/compute_norm_stats.py --config-name <cfg>`,
never inherited: the 100-task statistics describe a distribution the model never
sees. What matters most is that **both arms use identical stats**, so the
`sha256` of `norm_stats.json` is recorded in the run fingerprint alongside
`progress_filter`.
