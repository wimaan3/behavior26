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


---

## episode_index: the rule that could NOT be "don't renumber"

Preserving the global `episode_index` was the obvious counterpart to the
`task_index` rule. It is not implementable: **LeRobot v3 indexes episodes
positionally.**

```python
if ep_index >= len(self.episodes):
    raise IndexError(f"Episode index {ep_index} out of range. Episodes: {len(self.episodes)}")
ep = self.episodes[ep_index]          # row lookup into a datasets.Dataset
chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
```

`load_episodes` returns a `datasets.Dataset`, and `DatasetReader` separately
checks `requested_episodes = set(range(self._meta.total_episodes))`. A 50-episode
slice carrying global indices 4000-4049 therefore fails twice: `IndexError` on
video-path resolution, and a cache check that asks for 0-49 and finds none.

Dense zero-based episode indexing is structural.

### So the mapping rides as a COLUMN

`slice_task_dataset.py` renumbers `episode_index` densely and writes the original
as **`global_episode_index`**, registered in `meta/info.json` features (an
unregistered column makes the dataset unloadable -- the schema is built from that
dict, the same trap `progress` has).

A column rather than a sidecar file because **the mapping cannot be separated
from the rows it describes**: a sidecar can be paired with the wrong slice, lost
in a copy, or go stale after a re-slice. A column cannot, and a missing one fails
at merge, which is the property the design is buying.

`merge_progress_labels.py` gained `--labels-join-on` for the name asymmetry: the
data calls it `global_episode_index`, the Jetson sidecar -- produced against the
pristine corpus -- calls it `episode_index`. Same numbers, different name.
Required rather than auto-detected: auto-detection is how you silently merge
against the wrong root.

```bash
python scripts/slice_task_dataset.py --source <combined> \
    --task set_up_a_coffee_station_in_your_kitchen --out <tasks-root>

python scripts/merge_progress_labels.py --dataset-root <tasks-root>/<task> \
    --labels <jetson>/labels.parquet --out-root <arm-b-root> \
    --join-on global_episode_index frame_index \
    --labels-join-on episode_index frame_index
```

### The test that guards it

`tests/test_task_slicing.py::test_progress_labels_land_on_the_globally_correct_episodes`.

It labels **every** global episode -- a deliberate worst case. **Correction
(2026-09-14):** an earlier version of this section said the Jetson labels the
whole corpus. It does not: the sidecars are per task
(`origin/jetson/labels:labels/<task>/labels.parquet`), with global episodes
`task_index x 200 + i` -- coffee station 2000-2199 (199 labelled), shoe rack
4400-4599 (195 labelled).

That changes WHEN the bug is silent, not whether it matters. For the current pair a
wrong join on local 0..199 looks up turning_on_radio's range, finds nothing, and
fails loudly. It becomes silent once arm B concatenates sidecars covering global
0..N-1 -- plausible as k grows. The fixture covers that case so the test cannot pass
for the wrong reason.

Verified by mutation: pointing the join at the renumbered `episode_index` makes
the merge succeed and the **value** assertion fail. Re-run that mutation if the
test is ever refactored -- a green test here is only meaningful if it can go red.


## The third index trap: episode frame ranges

`meta/episodes` carries `dataset_from_index` / `dataset_to_index`, in the same space
as the data's `index` column. The reader clamps every delta-timestamp query into that
range and uses the result as a **row position** (`_absolute_to_relative_idx` is None
when all episodes load):

```python
query = max(ep_start, min(ep_end - 1, abs_idx + delta))
```

The first slicer renumbered `index` and copied the ranges unchanged -- so action
chunks would be drawn from the wrong frames: an IndexError for most tasks, but
**silently wrong action targets** for any task whose global offset lands inside the
slice. The synthetic fixture never exercised it because it had no range columns; it
does now, and the test was mutation-checked. `scripts/validate_slice.py` check 1c
verifies the same property on real data, frame by frame, at episode boundaries.

The slicer also now hard-links (or copies) every video file an episode references,
at its original path, and refuses to write a slice with a missing one.
