# Reward-shaping artifacts and the label pipeline for BEHAVIOR-1K

`labels.py` turns LeRobot parquet into per-frame progress targets. Everything else here
is the measurement that makes those labels trustworthy, plus the task selection that
follows from it. Nothing needs a GPU.

Two measurement scopes appear throughout, and the difference matters:

| Scope | What | Where |
|---|---|---|
| **sample** | 5,677 of 20,000 episodes (**28.4%**) -- the first parquet shard(s) of each task | `behavior1k_task_table.csv`, `behavior1k_episode_stats.csv`, `behavior1k_reward_map.json` |
| **full** | **all 20,000 episodes of all 100 tasks** | `full_corpus_remeasure.csv`, `full_corpus_lengths.csv`, `scope_delta.csv`, `bddl_audit.csv`, `task_shortlist.csv` |

Everything that drives task selection is now **full scope**. The sample-scope files are
retained only because the regression test pins against them.

`D` mostly survives the sampling but **not always** -- see warning 4 -- and **`phi0` does
not survive it at all**.

---

## The recipe

For each task, from the demo rewards alone:

1. **Measure the denominator.** `D = 1 / min|reward|` over all non-zero reward samples.
2. **Verify it.** Every observed magnitude must be an integer multiple of `1/D`.
3. **Repair rollback debits.** Zero any negative that would drive the running reward sum
   below zero. The demo collector rolls the simulator back and replays; the reward
   function then debits credit it never issued in this trace. No real predicate flip can
   take the sum below zero -- you cannot un-satisfy a unit that was never satisfied. A
   negative that leaves the sum at or above zero is a genuine flip-back and is **kept**.
4. **Build the potential.** With `cumsum_t` the repaired running sum:

   ```
   phi_t = D * (1 - cumsum_final + cumsum_t)
   ```

   `phi_t` is the number of goal units satisfied at frame `t`. Divide by `D` for
   normalised progress in `[0, 1]`. It ends at `D` (progress 1.0) on a successful demo.

### Validation gates

An episode is kept only if all of:

| Gate | Drop reason |
|---|---|
| terminates, and is not truncated | `not_terminated` / `truncated` |
| `phi0` in `[0, 1]` | `phi0_out_of_range` |
| `phi0 * D` is an integer | `phi0_not_integral` |
| `phi0 < 1` -- some headroom exists | `no_headroom` |
| progress stays in `[0, 1]` for **every** frame, not just `t=0` | `progress_out_of_range` |

### The identity that makes the audit possible

Because `phi0 * D` is always an integer:

```
phi0 * D        ==  goal units that NEVER fired during the demo
D * (1 - phi0)  ==  reward events actually observed
```

A demo that reaches the goal should credit every unit, so **a clean task has `phi0 = 0`**.
Any `phi0 > 0` on a successful demo means the reward under-fired. This is what caught
`putting_dishes_away_after_cleaning`.

---

## The label pipeline

```
python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
    --tasks turning_on_radio cook_bacon
python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
    --tasks-from analysis/reward/task_shortlist.csv
```

`--data-root` is a **local** LeRobot pull. Nothing in `labels.py` issues a network
request: the episode-to-shard map comes from `meta/episodes/`, on disk. That is
deliberate -- see the rate-limit note at the bottom.

Per task it writes:

```
<out>/<task>/labels.parquet    index, episode_index, frame_index, task_index,
                               progress (float32, [0,1]), satisfied_count (float32, [0,D])
<out>/<task>/manifest.json     D, magnitudes, episodes used/dropped, drop reasons,
                               mean/max phi0, frames, rollback debits zeroed,
                               non-monotonic episodes, provenance
<out>/<task>/REFUSED.json      written instead, when the task is refused
<out>/manifest.json            run summary: tasks emitted, tasks refused with reasons
```

`index` is the LeRobot dataset-global row index, so a data loader joins labels to frames
with a single key and no re-derivation. `(episode_index, frame_index)` is also emitted.

Both targets are present, as required: **`progress`** is normalised to `[0, 1]` for the
head; **`satisfied_count`** is the raw integer goal-unit count.

### Refusals

`labels.py` refuses a task rather than emitting labels it cannot stand behind:

| Condition | Why |
|---|---|
| not in the reward map | `D` has never been measured, and `D` must never be guessed |
| no reward signal / `D` unmeasurable | nothing to build a potential from |
| `reward_instrumentation != "ok"` | known-defective instrumentation |
| `phi0_mean > 0.05` | reward under-fires; labels would encode a scene already partly done |
| `valid_rate < 0.70` | too little of the task survives validation |
| `D` measured locally != `D` in the reward map | the pull disagrees with the map; refuse rather than pick one |
| a magnitude is not a multiple of `1/D` | `D` is wrong |
| `denominator_status == COLLAPSED`, with `--refuse-collapsed` | warning 3: `D` below the BDDL must-flip count. Off by default -- such a task is self-consistent and trainable, it just cannot demonstrate that shaping works. It always warns and always lands in the manifest. |

**`load_reward_map` overlays `full_corpus_remeasure.csv` on the sample-scope map where a
full measurement exists**, and every manifest records `measurement_scope`. Without this the
refusal gate is blind to exactly the defect warning 4 describes: reading the sample alone,
`installing_a_fax_machine` passes with `phi0 = 0`; reading the full corpus it is refused at
`phi0 = 0.1075`.

Refusals carry the evidence, not just a verdict:

```
REFUSED  putting_dishes_away_after_cleaning: reward_instrumentation = 'PROVEN_incomplete'
  -- mean phi0 = 0.9275 (full-200ep) with D = 14, so 13.0 of 14 goal units never fire
  even in demos that reach the goal; labels would encode a task already 93% complete at t=0
REFUSED  installing_a_fax_machine: reward instrumentation suspect -- mean phi0 = 0.1075
  (full-200ep) with D = 2, so 0.2 of 2 goal units never fire even in demos that reach the
  goal; labels would encode a task already 11% complete at t=0. Threshold is phi0 <= 0.05
REFUSED  rearranging_kitchen_furniture: no reward signal at all in the demos
  -- D is not measurable, so no progress label can be built
WARNING  make_microwave_popcorn: D = 1 but 2 BDDL predicates must flip -- part of the goal
  is invisible to the reward. Trainable, but it cannot demonstrate that shaping works.
```

### The recommended slate, end to end

```
$ python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
      --refuse-collapsed --tasks turning_on_radio hanging_pictures vacuuming_floors \
                                 installing_smoke_detectors cook_bacon
ok  turning_on_radio           D=1 eps=200/200 frames=429,928   mean_phi0=0.0000
ok  hanging_pictures           D=1 eps=198/200 frames=471,238   mean_phi0=0.0000
ok  vacuuming_floors           D=1 eps=199/200 frames=480,185   mean_phi0=0.0000
ok  installing_smoke_detectors D=1 eps=200/200 frames=513,754   mean_phi0=0.0000
ok  cook_bacon                 D=7 eps=197/200 frames=1,510,054 mean_phi0=0.0000
5 task(s) emitted, 0 refused
```

**3,405,159 labelled frames over 994 episodes**, every task `CONSISTENT`, `phi0 = 0`,
`full-200ep`. About 8 MB of parquet.

### Tests

`tests/test_labels.py` -- **31 tests, all against real parquet, no synthetic fixtures**
except two three-line arrays that pin the repair rule itself. Requires chunks 000, 008,
011, 046, 069 locally; skips cleanly otherwise.

The suite includes the under-instrumented task as an explicit refusal test
(`test_refuses_the_under_instrumented_reference_task`, and
`test_build_refuses_and_writes_no_labels_for_the_defective_task`), and a regression test
that reproduces `behavior1k_episode_stats.csv` -- per-episode `phi0`, repair count and
validity -- from raw parquet for `turning_on_radio`, `vacuuming_floors` and `cook_bacon`.
**255 of 255 episodes match exactly.** Disabling the repair fails 8 tests, so the pin has
teeth.

The same check over all five locally-held tasks (355 episodes, including
`putting_dishes_away_after_cleaning` and `rearranging_kitchen_furniture`) reproduces
`nnz`, `nneg`, `dropped`, `phi0`, `T` and terminal flags exactly, 355/355.

---

## Warning 1 -- `task_goal_terms.json` is not a source of `D`

It counts **top-level BDDL conjuncts**, which is not what the reward divides by. It agrees
with measured `D` on only **67 of 98** tasks. The failure is quantifiers: `picking_up_trash`
has 1 top-level conjunct (`forall` over three cans) but the reward uses `D = 3`.

**Always measure `D` from the reward trace. Retained for reference only.**

## Warning 2 -- the instrumentation defect

On some tasks the reward under-fires: the demo reaches the goal but reward never credits
every unit. **Validation does not catch this** -- such episodes pass every gate, because
the gates check internal consistency, not agreement with the BDDL goal. The signature is
`phi0`.

| Task | D | phi0 | Reward events (of D) | Status |
|---|---|---|---|---|
| `putting_dishes_away_after_cleaning` | 14 | 0.928 | 1.0 of 14 | **PROVEN incomplete** (full 200 ep) |
| `sorting_vegetables` | 13 | 0.703 | 3.9 of 13 | SUSPECTED (sample) |
| `assembling_gift_baskets` | 16 | 0.699 | 4.8 of 16 | SUSPECTED (sample) |
| `canning_food` | 10 | 0.638 | 3.6 of 10 | SUSPECTED (sample) |
| `rearranging_kitchen_furniture` | – | – | none | **BROKEN -- no signal** |
| `storing_food` | – | – | none | **BROKEN -- no signal** |

`putting_dishes_away_after_cleaning` is the proven case, and it holds on all 200 episodes:
`D = 14`, every episode fires reward **exactly once**, `phi0 = 13/14` in 200 of 200.

## Warning 3 -- `phi0 = 0` does NOT prove complete instrumentation

`phi0` measures firing *relative to D*. If `D` was collapsed below the true goal size
before measurement, `phi0 = 0` merely confirms `D` and the observed events agree. It says
nothing about whether `D` matches the BDDL goal, and `phi0` is blind to this.

Comparing measured `D` against BDDL predicates that must flip (`../census/`):
**32 tasks** have `D` < must-flip count, and **14 of them are `phi0 = 0`, valid, and would
otherwise rank** -- listed in `task_shortlist.csv` under
`tier = excluded_collapsed_denominator`. `sorting_bottles_cans_and_paper` is the clearest:
16 predicates must flip, `D = 3`, and 3 does not even divide 16.

**A task is trustworthy only when `phi0 = 0` AND `D` equals the must-flip count.**
`denominator_status` reports this as `CONSISTENT` / `COLLAPSED` / `UNDER_FIRES`.

## Warning 4 -- the 28.4% sample gets `D` wrong on one task and `phi0` wrong on many (updated: full corpus)

`behavior1k_task_table.csv` and friends were measured over the first parquet shard(s) of
each task: **5,677 of 20,000 episodes**. All 100 tasks have now been re-measured over all
20,000 episodes (`measure_corpus.py` against the reward mirror, see `fetch_rewards.py`).

**`D` changed on one task**, and it is the failure mode that matters:

| Task | sample `D` | full `D` | why |
|---|---|---|---|
| `wash_dog_toys` | 3 | **6** | its 19-episode sample contained only the magnitudes `1/3` and `2/3` |

`D = 1/min|reward|`, so it is fixed by the single smallest magnitude anywhere in the task.
Across all 200 `wash_dog_toys` episodes the magnitude `1/6` occurs **exactly once**
(`1/2` also occurs exactly once). One reward sample in the whole task sets its
denominator. `D` is therefore **not** safe to measure on a sample, on any task, and the
99 tasks where it did not move are luck rather than evidence. Every magnitude on every
task is an integer multiple of its `1/D`: **integrality passes on all 98 tasks with a
measurable `D`.**

`phi0` moves on far more tasks, always upward -- a sample flatters the instrumentation:

| Task | sample `phi0` | sample eps | full `phi0` | episodes with `phi0 > 0` (of 200) |
|---|---|---|---|---|
| `putting_dirty_dishes_in_sink` | 0.0 | | **0.2539** | 98 |
| `dispose_of_batteries` | 0.0 | | **0.2513** | 99 |
| `make_gift_bags_for_baby_showers` | 0.1449 | | **0.3407** | 125 |
| `installing_a_scanner` | 0.0 | | **0.1667** | 66 |
| `packing_meal_for_delivery` | 0.0 | | **0.1658** | 64 |
| `composting_waste` | 0.0 | | **0.1625** | 65 |
| `installing_a_fax_machine` | 0.0 | 62 | **0.1075** | 43 |

Seven tasks read `phi0 = 0` on their sample and are under-firing on the full corpus.
`installing_a_fax_machine` was rank 9 of an earlier shortlist on the strength of
`phi0 = 0`.

**The null baseline** -- task-equal mean `phi0` over tasks whose instrumentation is `ok`
-- moves from **0.0932 (sample) to 0.1293 (full corpus)**, n = 94 both times.

## Warning 5 -- zero-headroom episodes (new)

One `vacuuming_floors` episode and two `hanging_pictures` episodes **terminate having
fired no reward at all**: `phi0 = 1`, so progress is the constant `1.0` for every frame.
The label would teach that a freshly-reset scene is already complete. These pass range
(1 is in `[0,1]`) and integrality (`1*D` is integral), so nothing in the original recipe
rejected them. `labels.py` drops them as `no_headroom`. They are invisible in the sample.

## Warning 6 -- non-monotonic progress

A genuine predicate flip-back survives repair by design, so progress can dip mid-episode.
This is not rare on multi-unit tasks: **44 of 197** valid `cook_bacon` episodes,
**146 of 200** `chop_an_onion`, **131 of 188** `make_rose_centerpieces`. Every manifest
reports `episodes_nonmonotonic`. `labels.py` does not smooth it -- the dips are real
predicate state -- but a progress head trained with a monotonicity prior will fight them.
Decide this before training, not after.

---

## Task selection

Budget is 2–8 tasks at ~64 GPU-hours each. Criteria, in the stated priority order:

1. **`phi0 = 0` in every episode** (necessary, not sufficient -- warning 3).
2. **Short mean episode length** -- eval timeout is `1.5x` the task's own mean demo
   length, so a short task is far cheaper to evaluate.
3. **`valid_rate >= 0.70`**.
4. **Enough episodes** -- all 200 for every task below.

Plus the task-3 exclusion: a `COLLAPSED` denominator removes a task from the primary tier.

### Why length dominates

Full-corpus mean demo length runs **2,150 to 27,000+ frames** (corpus mean 10,546 over all
20,000 episodes). Eval cost is essentially linear in the timeout, so task choice alone
swings the eval bill by an order of magnitude. `rel_eval_cost` is normalised to the corpus
mean: the top pick costs **0.20x** an average task. The whole recommended four costs
**0.90x a single average task**.

### Ranked shortlist

Ranks 1–11 are measured on **all 200 episodes** with this pipeline. Full detail in
`task_shortlist.csv`; the audit columns are in `bddl_audit.csv`.

| # | Task | D | Mean len | Timeout | Rel cost | Valid | phi0>0 | Non-mono | Denominator |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `turning_on_radio` | 1 | 2,150 | 3,224 | 0.20x | 1.000 | 0 | 1 | CONSISTENT |
| 2 | `hanging_pictures` | 1 | 2,387 | 3,581 | 0.23x | 0.990 | 0 | 0 | CONSISTENT |
| 3 | `vacuuming_floors` | 1 | 2,412 | 3,618 | 0.23x | 0.995 | 0 | 0 | CONSISTENT |
| 4 | `installing_smoke_detectors` | 1 | 2,569 | 3,853 | 0.24x | 1.000 | 0 | 0 | CONSISTENT |
| 5 | `scrubbing_bathroom_floor` | 1 | 3,154 | 4,731 | 0.30x | 1.000 | 0 | 0 | CONSISTENT |
| 6 | `make_cabinet_doors` | 1 | 3,442 | 5,164 | 0.33x | 1.000 | 0 | 0 | CONSISTENT |
| 7 | `clean_a_keyboard` | 1 | 3,876 | 5,814 | 0.37x | 1.000 | 0 | 0 | CONSISTENT |
| 8 | `attach_a_camera_to_a_tripod` | 1 | 3,912 | 5,867 | 0.37x | 1.000 | 0 | 0 | CONSISTENT |
| 9 | `clean_a_trumpet` | 1 | 5,307 | 7,961 | 0.50x | 0.990 | 0 | 0 | CONSISTENT |
| 10 | `store_honey` | 1 | 6,767 | 10,150 | 0.64x | 1.000 | 0 | 8 | CONSISTENT |
| 11 | **`cook_bacon`** | **7** | 7,680 | 11,519 | 0.73x | 0.985 | 0 | 44 | CONSISTENT |
| 12 | `cook_a_frozen_pie` | 2 | 8,668 | 13,002 | 0.82x | 0.855 | – | – | CONSISTENT *(sample only)* |

**Recommended 4 if the budget is tight:** `turning_on_radio`, `hanging_pictures`,
`vacuuming_floors`, `installing_smoke_detectors` -- all `D = 1`, `phi0 = 0` on all 200
episodes, validity >= 0.99, `CONSISTENT`, and together **0.90x the eval cost of one
average task**. Varied in skill: a toggle, two `attached()` placements, and surface
cleaning.

**Recommended 5th: `cook_bacon`.** See below.

### Medium-D: state the tradeoff, then note it mostly is not one

The `phi0 = 0` set skews hard to `D = 1`: 30 tasks qualify on criteria 1, 3 and 4, warning 3
removes 14, and of the 16 that remain almost all are binary
rewards, where a shaping target carries almost no information -- progress is 0 until it
is 1. We want at least one task with real progress structure.

**`cook_bacon` is the only `D >= 4` task in the entire corpus with `phi0 = 0`.** Every
other task with `D >= 4` is `UNDER_FIRES` or `COLLAPSED` -- verified across all 98 tasks
with a measurable `D` in `bddl_audit.csv`. It has `D = 7` (6 bacon slices cooking
independently, plus a fridge-closed literal already true at init, so 6 units are real
work), `phi0 = 0` in 200 of 200 episodes, validity 0.985, `CONSISTENT`, and at **0.73x**
it is cheaper than previously believed (its sampled mean length overstated by 10%).

So the tradeoff is far smaller than the earlier analysis claimed: `cook_bacon` ranks 11 in
the primary tier on its own merits. It costs about as much as three of the top four
together, and its one real cost is warning 6 -- 44 of 197 episodes have non-monotonic
progress.

The two next-best `D >= 4` candidates are genuinely compromised at full scale, and are
listed as `medium_D` in the shortlist for completeness, **not** as recommendations:

| Task | D | Rel cost | Valid | phi0 (full) | eps with phi0>0 | Non-mono | Verdict |
|---|---|---|---|---|---|---|---|
| `make_rose_centerpieces` | 4 | 0.43x | 0.940 | 0.136 | 81/200 | 131 | UNDER_FIRES -- worse than the sample said |
| `chop_an_onion` | 4 | 0.61x | 1.000 | 0.065 | 48/200 | 146 | UNDER_FIRES -- 73% non-monotonic |

**Suggested 5-task slate:** the recommended 4 plus `cook_bacon`. Total **1.63x** the eval
cost of a single average task. If a sixth is wanted, `scrubbing_bathroom_floor` (0.30x).

### The organizers' two suggestions

- **`turning_on_radio` -- rank 1.** Confirmed excellent, and it improves on the fuller
  measurement: shortest task in the corpus at 2,150 frames (the sample overstated it by
  8%), validity 1.000 across all 200 episodes, `D = 1` matching its single `toggled_on`
  literal. Take it.
- **`make_microwave_popcorn` -- excluded.** It is short (3,238) and `phi0 = 0` with
  validity 1.000 on all 200 episodes, so on the stated criteria it looks clean. The audit
  says otherwise: its goal is **two** literals -- `real(cooked__popcorn)` and
  `contains(popcorn__bag, cooked__popcorn)` -- and `D = 1`. Both must flip; reward fires
  once. `COLLAPSED`. Per task 3 it is excluded from the shortlist. It would train
  (self-consistent, full headroom) but half its goal is invisible to the reward, so it can
  never demonstrate that shaping works.

---

## Instrumentation audit (task 3)

Method: read each task's BDDL goal literals from `../census/goal_census_detail.json`, count
those not already true at init (**must flip**), and compare against reward events actually
observed, `D * (1 - phi0)`. Literals with unknown init state are counted as needing to
flip. Full table for all 98 tasks with a measurable `D` in `bddl_audit.csv`.

| Task | D | Must flip | Literals | Reward events | Verdict |
|---|---|---|---|---|---|
| `turning_on_radio` | 1 | 1 | 1 | 1.00 | **clean** |
| `hanging_pictures` | 1 | 1 | 1 | 1.00 | **clean** |
| `vacuuming_floors` | 1 | 1 | 1 | 1.00 | **clean** |
| `installing_smoke_detectors` | 1 | 1 | 1 | 1.00 | **clean** |
| `scrubbing_bathroom_floor` | 1 | 1 | 1 | 1.00 | **clean** |
| `make_cabinet_doors` | 1 | 1 | 1 | 1.00 | **clean** |
| `clean_a_keyboard` | 1 | 1 | 1 | 1.00 | **clean** |
| `attach_a_camera_to_a_tripod` | 1 | 1 | 1 | 1.00 | **clean** |
| `clean_a_trumpet` | 1 | 1 | 1 | 1.00 | **clean** |
| `store_honey` | 1 | 1 | 1 | 1.00 | **clean** |
| `cook_bacon` | 7 | 6 | 7 | 7.00 | **clean** (1 literal already true at init) |
| `cook_a_frozen_pie` | 2 | 2 | 2 | 2.00 | **clean** *(sample only)* |
| `make_microwave_popcorn` | 1 | 2 | 2 | 1.00 | **do not use** -- `COLLAPSED`, excluded |
| `installing_a_fax_machine` | 2 | 2 | 2 | 1.78 | **do not use** -- `UNDER_FIRES` at full scale, excluded |
| `make_rose_centerpieces` | 4 | 4 | 4 | 3.46 | **suspect** -- 81/200 episodes under-fire |
| `chop_an_onion` | 4 | 4 | 4 | 3.74 | **suspect** -- 48/200 episodes under-fire |
| `sweeping_garage` | 2 | 2 | 2 | 1.98 | **suspect** -- 4/200, marginal; dropped from the list |
| `putting_dishes_away_after_cleaning` | 14 | 8 | 10 | 1.02 | **do not use** -- reference defect |

**Corpus-wide result, all 100 tasks on all 20,000 episodes:**

| Verdict | Tasks |
|---|---|
| `CONSISTENT` -- `D` == must-flip count and every episode credits all `D` units | **13** |
| `UNDER_FIRES` -- `D` right, successful demos credit fewer than `D` | **54** |
| `COLLAPSED` -- `D` below the must-flip count | **31** |
| no reward signal at all | **2** |

**Only 13 of 100 tasks are safe to build a progress label from**, and all 13 are in the
primary tier of `task_shortlist.csv`. 26 tasks have `phi0 = 0` in every episode, but 13 of
those are `COLLAPSED` -- warning 3 exactly.

---

## Files

| File | What it is |
|---|---|
| `labels.py` | **The label pipeline.** Parquet in, per-frame `progress` + `satisfied_count` out, with refusals. |
| `tests/test_labels.py` | 31 tests against real parquet, including the refusal case and the corpus regression pin. |
| `measure_corpus.py` | Re-measures `D`, `phi0`, validity on **all 200** episodes of every locally-held task; also emits corpus-wide lengths. |
| `build_shortlist.py` | Regenerates `task_shortlist.csv` and `bddl_audit.csv`. |
| `og_state_decoder.py` | Decodes OmniGibson flat state vectors into named object poses/joints via the `scene_file` attr in each HDF5. Handles the assisted-grasp sentinel block and carry-forward of absent objects. |
| `behavior1k_reward_map.json` | Per task: `D`, `phi0_mean`, counts, validity, `reward_instrumentation`, `usable_for_labels`. The file `labels.py` loads. **Sample-scope.** |
| `behavior1k_task_table.csv` | 100 rows, per-task measurement detail. **Sample-scope.** |
| `behavior1k_episode_stats.csv` | 5,677 rows, one per episode. **Sample-scope.** The regression test pins against this. |
| `full_corpus_lengths.csv` | All 100 tasks, mean/median/p90 length over all 20,000 episodes, plus eval timeout and relative cost. |
| `full_corpus_remeasure.csv` | 21 tasks re-measured over all 200 episodes with this pipeline. |
| `task_shortlist.csv` | Training-candidate ranking (task 2), with the audit columns. |
| `bddl_audit.csv` | Task-3 audit for all 98 tasks with a measurable `D`. |
| `lerobot_sweep_ep_stats.csv` | 316-episode sweep with each validation gate broken out. |
| `lerobot_reward_ep_stats.csv` | The original 12-task reward reconnaissance. |
| `reward_denominators.json` | The 12 tasks used to derive the recipe: observed `D` vs top-level conjuncts vs quantifier-expanded count. |
| `task_goal_terms.json` | Top-level BDDL conjunct counts. **Not a source of `D`** -- warning 1. |

---

## Dataset provenance

The demo repo `behavior-1k/2026-challenge-demos` was patched twice:

| Commit | Date | What |
|---|---|---|
| `e6c97564` | 2026-07-28 | base `qvel` states |
| `4f50b447` | **2026-08-05** | arm, trunk and gripper `qvel` in `observation.state`, plus `meta/stats.json` |

`4f50b447` is still `main` -- no commits since. Every measurement here was taken from a
pull at that revision. Both patches touch `observation.state` only; `next.reward`,
`next.terminated` and `next.truncated` are untouched by either, so the reward analysis
would have been valid pre-patch too. A policy trained on pre-patch `observation.state`
would not be.

## Do not let `compute_norm_stats.py` resolve episodes over the network

`compute_norm_stats.py --config-name pi05_b1k` issues a **per-episode** resolver request
across every task and trips HuggingFace's 5,000-requests-per-5-minutes limit. Point the
**dataset config** at a local path before running it. Organizer-confirmed.

`labels.py` and `measure_corpus.py` are already immune: they resolve every episode from
`meta/episodes/` on disk and never call the hub.
