# Slice validation on real data — CPU pod, 2026-09-14

Pod `cnxjzvamszi82z`, cpu3g 4 vCPU / 16 GB, EU-RO-1, $0.16/hr, 40 GB container
disk (CPU pods cap at 40). Terminated. **Cost ≈ $0.09.** No GPU time.

Real LeRobot (`wensi-ai/lerobot@release/b1k`, torch 2.11.0), real chunk-010
(15 GB, 200 episodes), real Jetson sidecar (`origin/jetson/labels`, 3.6 MB).

## All four checks PASS

| check | result |
|---|---|
| **1a** LeRobotDataset opens the slice | `len=1,253,243` rows, meta 200 episodes == data 200 |
| **1b** video paths resolve | 200 episodes x **6 video keys**, none missing |
| **1c** action chunks come from the right frames | 9 probes x horizon 16 at episode boundaries, **0 mismatches** |
| **1d** a video frame decodes | rgb 3x480x480 / 3x720x720, depth 1x480x480 — real pixels |
| **2** `global_episode_index` survives | registered, present in the LeRobot item, one global per local, **local 0..199 -> global 2000..2199** |
| **3** every `task_index` resolves | `[10] -> ['set_up_a_coffee_station_in_your_kitchen']`, table has all 100 |
| **4** labels land on globally correct episodes | **1,247,890 rows, 0 value mismatches, 0 sidecar-task mismatches**, 199/200 episodes kept |

Check 1c is the real-data version of the `dataset_from_index` trap fixed earlier
the same day: action chunks compared frame by frame against raw rows at the start,
middle and end of three episodes. Check 4 is the rewritten arm-B poisoning test
against the actual sidecar rather than a fixture — the strongest form available
before a training run.

The one dropped episode is **111** (5,353 unlabelled frames); the sidecar covers
199 of the chunk's 200 episodes. `--drop-unlabelled` removed it whole, which is
what AB_PROTOCOL 3.6 requires of both arms.

## Two defects found, one fixed

**FIXED — `--drop-unlabelled` lost the episode column.** The merge selected only
the join columns out of each parquet, so once the join moved to
`global_episode_index` the dropper had no `episode_index` to work with and the
run died. The synthetic test had never used the flag. Now covered.

**OPEN — the merged (arm B) root is not loadable by LeRobot v3.**
`merge_progress_labels.py` deliberately does not renumber episodes, which was
right for v2 but breaks v3's positional indexing. Measured on the merged root:

```
info.json      total_episodes=199   total_frames=1,247,890
meta/episodes  200 rows            <- still lists the dropped episode
data           199 episodes        <- gap at episode_index 111
data index     0..1,253,242 for 1,247,890 rows   <- no longer dense
```

So `self.episodes[ep_index]` is off by one for every episode after 111, the
`dataset_from_index` / `dataset_to_index` ranges no longer match the rows, and
`LeRobotDataset` falls through to a Hub lookup for a repo that does not exist
(`RepositoryNotFoundError: 401`, or `OfflineModeIsEnabled` with `HF_HUB_OFFLINE=1`
— the same failure wearing two different masks).

**Arm B could not train on this root today**, and arm A could, which is exactly
the asymmetry we have been hunting all session. It is not a data-integrity
problem — the labels themselves are provably correct, per check 4 — but a
compaction problem: after dropping episodes the merge must renumber
`episode_index` densely, rewrite `index` and the episode ranges, and prune
`meta/episodes`, preserving `global_episode_index` so labels stay traceable.

That is the next piece of local work, with tests, before the GPU pod.
