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

### What changed in the current revision

The progress label is now anchored at the **start** of the episode rather than the end.
The previous recipe assumed every demo finished the task and shifted the curve until it
did, which relabelled a demo that stopped one unit short as a demo that started one unit
ahead. Consequences, all detailed below:

- **The corpus is not 87% broken. That was our audit method.** The old table read 13
  `CONSISTENT` and 54 `UNDER_FIRES`, and the 54 was terminal-anchored `phi0` counting
  truncated demos as missing reward. Re-measured, the under-firing class is **4 tasks**,
  not 54: `putting_dishes_away_after_cleaning`, `assembling_gift_baskets`,
  `sorting_vegetables`, `hiding_Easter_eggs`, each with **0 of 200** episodes reaching the
  goal. The other 49 have sound instrumentation and demos that stop a unit or two short.
  **Tasks safe to build a progress label from: 63, not 13.** `installing_a_fax_machine` is
  no longer excluded; `wash_dog_toys` is `CONSISTENT` with its two "missing" units
  correctly identified as satisfied at reset. `COLLAPSED` (31) and the 2 no-signal tasks
  are unaffected -- those findings were read from the BDDL, not from `phi0`, and they
  stand.
- The null baseline drops from **0.1293 to 0.0466**, and the eval-side null is **0**.
- `task_shortlist.csv` carries `gradient_score` and is no longer truncated to 12 rows.
  **12 of the 14 `CONSISTENT` tasks have `gradient_score = 0`** -- including `cook_bacon`,
  whose D=7 lands six units in a single frame.
- `labels.parquet` **keeps its exact schema**: same columns, order, dtypes, join keys and
  `[0,1]` range. Only the values change. `test_emitted_label_schema_is_exactly_as_pinned`
  holds it there.

---

## The recipe

For each task, from the demo rewards plus the BDDL `:init` block:

1. **Measure the denominator.** `D = 1 / min|reward|` over all non-zero reward samples.
2. **Verify it.** Every observed magnitude must be an integer multiple of `1/D`.
3. **Read the goal state at reset from the BDDL.** `s0` is the number of goal literals
   the census marks `initially_true`. Literals marked *unknown* are **not** counted --
   see the note on the residual below.
4. **Repair rollback debits.** Zero any negative that would drive the running reward sum
   below `-s0/D`. The demo collector rolls the simulator back and replays; the reward
   function then debits credit it never issued in this trace. No real predicate flip can
   take the sum below the number of units that were true at reset -- you cannot
   un-satisfy a unit that was never satisfied. A negative that stays at or above the
   floor is a genuine flip-back and is **kept**.
5. **Cap `s0` against the trace.** A trace that credits `k` units cannot have started
   with more than `D - k` already satisfied, so `s0 = min(s0, D - k)`. The census
   describes the scene, not the episode, and it is sometimes wrong about it: 44 of 200
   `cook_bacon` episodes earn credit for **closing** the refrigerator with no preceding
   debit for opening it, so in those episodes the door was open at reset and
   `not open(...) initially_true` does not hold. Lowering `s0` lowers the repair floor,
   so steps 4 and 5 iterate to a fixed point.
6. **Build the potential.** With `cumsum_t` the repaired running sum:

   ```
   satisfied_t = s0 + D * cumsum_t
   ```

   `satisfied_t` is the number of goal units true at frame `t`. Divide by `D` for
   normalised progress in `[0, 1]`.

**The label is anchored at the start of the episode, not the end.** The previous recipe,
`phi_t = D * (1 - cumsum_final + cumsum_t)`, assumed every demo finished the task and
shifted the whole curve until it did. A demo that stopped one unit short was relabelled
as having *started* one unit ahead. That single assumption produced the entire
`UNDER_FIRES` band -- see warning 2. It is still reachable as `anchor="terminal"`, which
is what `behavior1k_episode_stats.csv` was measured with, and the two anchors agree
exactly on any task with nothing true at reset whose demos all finish.

### Validation gates

An episode is kept only if all of:

| Gate | Drop reason |
|---|---|
| terminates, and is not truncated | `not_terminated` / `truncated` |
| `phi0` in `[0, 1]` | `phi0_out_of_range` |
| `phi0 * D` is an integer | `phi0_not_integral` |
| the trace credits no more than `D` units | `over_credited` |
| progress is not constant for the whole episode | `no_headroom` |
| progress stays in `[0, 1]` for **every** frame, not just `t=0` | `progress_out_of_range` |

`over_credited` is the rollback-replay signature: the same unit credited twice. It
replaces the old `phi0_out_of_range` reading of the same three `cook_bacon` episodes.
`no_headroom` now means *the label never changes*, which catches both an episode that
fired nothing (the constant 0) and the old constant-1.0 case.

### The identities that make the audit possible

```
phi0 * D             ==  goal units already true at reset      (a scene property)
D * (1 - peak)       ==  goal units the reward NEVER credits   (an instrument defect)
D * (peak - final)   ==  units earned and then lost before the end
demo_completion_rate ==  share of valid episodes reaching progress 1.0
```

The middle two are what terminal-anchored `phi0` was silently summing. **A clean task
has `never_credited = 0` and `demo_completion_rate = 1`.** A task where no episode ever
completes is an instrument defect -- this is what caught
`putting_dishes_away_after_cleaning`. A task where some episodes complete and others
stop short has sound instrumentation and truncated demos, and is labelled correctly
episode by episode.

### Open design question: init-anchored labels vs the Q metric

**Not decided. Decide before training, not after.**

The labels above count predicates *currently true*. `Q` counts predicates *flipped since
reset*. From `OmniGibson/omnigibson/metrics/task_metric.py`:

```python
sum(int(not initially_true and pred.evaluate(...)) for pred, initially_true in ...)
/ len(option)
```

A literal true at reset scores **zero forever** while still occupying a slot in the
denominator. So the two targets differ by exactly the reset offset, and a policy can never
earn that offset back:

| Task | Init-anchored label runs | Q-aligned label would run (`1 - offset`) | Offset |
|---|---|---|---|
| `putting_shoes_on_rack` | 0.000 → 1.0 | 0.000 → 1.0 | **0.000** |
| `set_up_a_coffee_station_in_your_kitchen` | 0.165 → 1.0 | 0.000 → 0.835 | **0.165** |
| `outfit_a_basic_toolbox` | 0.266 → 1.0 | 0.000 → 0.734 | **0.266** |
| `cook_bacon` | 0.117 → 1.0 | 0.000 → 0.883 | 0.117 |
| `wash_dog_toys` | 0.333 → 1.0 | 0.000 → 0.667 | 0.333 |

**Option A -- keep the init anchor (what `labels.py` emits today).** `progress` is the
fraction of the goal currently satisfied. It is what the reward stream literally measures,
it is the quantity the repair rule and the `s0` cap are defined against, and a constant
offset is irrelevant to potential-based shaping, where only `gamma*Phi(s') - Phi(s)`
enters. The cost is that a regression head trained on it predicts a number that is not `Q`
and is systematically above it by the offset.

**Option B -- subtract the offset, so the head predicts earned progress.**
`progress_q = (satisfied_t - s0) / D`, clipped at 0. Directly comparable to `Q`, starts at
0 on every task, and makes the eval-side null of 0 the natural floor. The cost is that it
is not exactly `Q` either: the reward re-credits an initially-true literal that is broken
and then restored, and a scalar reward cannot say which literal that was, so the estimate
can drift above the true earned count by up to `s0` units. `cook_bacon` is the worked
example -- closing the refrigerator earns `+1/7` in the trace and nothing at all in `Q`.

The offset is recorded per task as `progress_offset_at_reset` in `task_shortlist.csv` and
as `mean_phi0` / `initially_true_literals` in every manifest, so either target can be
built from the emitted `satisfied_count` without re-running the pipeline. **On
`putting_shoes_on_rack` the two options are identical**, which is one more reason to make
it the primary signal task.

### The residual: unknown initial states

The census marks a literal `initially_true: None` when the BDDL `:init` block does not
state it -- mostly kinematic relations. Those are deliberately **not** counted toward
`s0`. Counting them lets a rollback debit masquerade as a broken initially-true literal,
which on a `D = 1` task pins progress at a constant 1.0 and throws the episode away; it
cost 104 of 200 `re_shelving_library_books` episodes before this was tightened.

The cost of not counting them is a bounded over-credit in the other direction: if a
literal really is true at reset and the robot breaks it, the debit is repaired away and
the restoring credit is kept, inflating the trace by up to `initially_true_unknown`
units. Both counts are in every manifest and in `task_shortlist.csv`, and
`rollback_rewards_zeroed` is the tell -- a task with many unknowns *and* a repair on
nearly every episode is the one to check against a live reset first
(`thawing_frozen_food`: 6 unknown, 198 repairs over 200 episodes).

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
                               reset offset and the literal counts behind it, units
                               never credited, units lost at the end, demo completion
                               rate, progress_structure, frames, rollback debits
                               zeroed, non-monotonic episodes, provenance
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
| `demo_completion_rate == 0` | no demo anywhere in the task reaches the whole goal, so the reward cannot express completion. Replaces the old `phi0_mean > 0.05` gate, which could not tell an under-firing reward from a demo that stopped a step early |
| `phi0_init_mean > --max-phi0` | optional and **off by default**: the reset offset is a property of the scene, not a defect. `wash_dog_toys` starts at 0.333 because two of its six literals are satisfied at reset and never broken |
| `valid_rate < 0.70` | too little of the task survives validation |
| `D` measured locally != `D` in the reward map | the pull disagrees with the map; refuse rather than pick one |
| a magnitude is not a multiple of `1/D` | `D` is wrong |
| `denominator_status == COLLAPSED`, with `--refuse-collapsed` | warning 3: `D` below the BDDL must-flip count. Off by default -- such a task is self-consistent and trainable, it just cannot demonstrate that shaping works. It always warns and always lands in the manifest. |

**`load_reward_map` overlays `full_corpus_remeasure.csv` on the sample-scope map where a
full measurement exists**, and every manifest records `measurement_scope`. Without this the
refusal gate reads a 28.4% sample -- see warning 4. A map that predates the init anchor
carries no `demo_completion_rate`, and the gate falls back to terminal-anchored `phi0`
with a message saying so, rather than passing everything silently.

Refusals carry the evidence, not just a verdict:

```
REFUSED  putting_dishes_away_after_cleaning: reward_instrumentation = 'PROVEN_incomplete'
  (full-200ep)
REFUSED  rearranging_kitchen_furniture: no reward signal at all in the demos
  -- D is not measurable, so no progress label can be built
WARNING  make_microwave_popcorn: D = 1 but 2 BDDL predicates must flip -- part of the goal
  is invisible to the reward. Trainable, but it cannot demonstrate that shaping works.
```

With the instrumentation flag cleared, the completion gate refuses the reference defect
on its own evidence:

```
no demo reaches the whole goal: 0.0% of valid episodes end at progress 1.0 (full-200ep),
with 12.99 of 14 goal units never credited anywhere in the mean episode. The reward
cannot express completion of this task
```

**`installing_a_fax_machine` is no longer refused.** Terminal-anchored `phi0` read 0.1075
across 200 episodes and the task was excluded as under-instrumented. Init-anchored, 78.5%
of its episodes reach progress 1.0 and the rest stop one unit short: nothing is wrong with
its reward, its demos truncate. It is `ENDS_SHORT`, and there is a test pinning the
reversal.

### The recommended slate, end to end

```
$ python -m analysis.reward.labels --data-root ~/behavior-data --out labels/ \
      --refuse-collapsed --tasks turning_on_radio vacuuming_floors \
                                 installing_smoke_detectors \
                                 set_up_a_coffee_station_in_your_kitchen \
                                 putting_shoes_on_rack
ok  turning_on_radio           D=1  eps=200/200 frames=429928  offset@reset=0.0000 complete=100% gradient=0.00/maxstep=1.00
ok  vacuuming_floors           D=1  eps=199/200 frames=480185  offset@reset=0.0000 complete=100% gradient=0.00/maxstep=1.00
ok  installing_smoke_detectors D=1  eps=200/200 frames=513754  offset@reset=0.0000 complete=100% gradient=0.00/maxstep=1.00
ok  set_up_a_coffee_station... D=6  eps=199/200 frames=1247890 offset@reset=0.1650 complete=32%  gradient=0.80/maxstep=0.17  [ENDS_SHORT]
ok  putting_shoes_on_rack      D=10 eps=195/200 frames=1503533 offset@reset=0.0000 complete=33%  gradient=0.80/maxstep=0.10  [ENDS_SHORT]

5 task(s) emitted, 0 refused
```

**4,175,290 labelled frames over 993 episodes**, `full-200ep`, about 12 MB of parquet.
Three controls with no gradient and two tasks that have one. The per-task line reports
what now matters: where the label starts, how many demos finish, and whether the curve is
a staircase or a cliff.

`--data-root` can be a reward-only mirror. `labels.py` reads six columns
(`index`, `episode_index`, `frame_index`, `next.reward`, `next.terminated`,
`next.truncated`), so the 799 MB mirror built by `fetch_rewards.py` is enough to label all
100 tasks without pulling 2 TB of video.

### Tests

`tests/test_labels.py` -- **45 tests**, mostly against real parquet; the synthetic ones
are short arrays that pin the repair rule, the `s0` cap and the structure metrics, where a
hand-built trace states the intent more clearly than a 9,000-frame episode. Requires
chunks 000, 008, 011, 046, 069, 089 locally; skips cleanly otherwise.

The suite includes the under-instrumented task as an explicit refusal test
(`test_refuses_the_under_instrumented_reference_task`,
`test_still_refuses_when_no_demo_in_the_task_ever_completes` and
`test_build_refuses_and_writes_no_labels_for_the_defective_task`), and a regression test
that reproduces `behavior1k_episode_stats.csv` -- per-episode `phi0`, repair count and
validity -- from raw parquet for `turning_on_radio`, `vacuuming_floors` and `cook_bacon`.
**255 of 255 episodes match exactly.** That pin now names `anchor="terminal"` explicitly,
because that is the recipe the CSV was measured with; it passes unchanged on
`turning_on_radio` and `vacuuming_floors` under either anchor.

Three tests exist because of the anchor change specifically:

- `test_emitted_label_schema_is_exactly_as_pinned` -- column names, order and physical
  types of `labels.parquet`, plus `progress` in `[0,1]` and
  `satisfied_count == progress * D`. The anchor changes the values in those two columns
  and must not change anything else.
- `test_join_keys_are_unique_and_dense` -- `(episode_index, frame_index)` is a primary key
  and `frame_index` is a dense `0..n-1` per episode.
- `test_the_task_that_only_under_fired_at_full_scale_is_now_accepted` -- pins the
  `installing_a_fax_machine` reversal, so a future change that silently re-refuses it
  fails loudly.

The same check over all five locally-held tasks (355 episodes, including
`putting_dishes_away_after_cleaning` and `rearranging_kitchen_furniture`) reproduces
`nnz`, `nneg`, `dropped`, `phi0`, `T` and terminal flags exactly, 355/355.

---

## Warning 1 -- `task_goal_terms.json` is not a source of `D`

It counts **top-level BDDL conjuncts**, which is not what the reward divides by. It agrees
with measured `D` on only **67 of 98** tasks. The failure is quantifiers: `picking_up_trash`
has 1 top-level conjunct (`forall` over three cans) but the reward uses `D = 3`.

**Always measure `D` from the reward trace. Retained for reference only.**

## Warning 2 -- most of the "instrumentation defect" was the anchor

On some tasks the demo reaches the goal but the reward never credits every unit. That is
real, and it is rare. What the terminal anchor reported instead was mostly something
else: a demo that stopped one or two units short of the goal, relabelled as a demo that
had started that far ahead.

Splitting the shortfall into the two things it was summing, over all 200 episodes of
every task:

| Task | D | never credited | lost at end | demos completing | verdict |
|---|---|---|---|---|---|
| `putting_dishes_away_after_cleaning` | 14 | **12.99** | 0.00 | **0 of 200** | `DEAD_UNITS` -- reference defect, still refused |
| `hiding_Easter_eggs` | 9 | 3.40 | 0.00 | 0 of 200 | `DEAD_UNITS` |
| `chop_an_onion` | 4 | 0.26 | 0.00 | 152 of 200 | `ENDS_SHORT` -- instrumentation is fine |
| `installing_a_fax_machine` | 2 | 0.22 | 0.00 | 157 of 200 | `ENDS_SHORT` -- **no longer excluded** |
| `wash_dog_toys` | 6 | **0.00** | 0.00 | 199 of 199 | `CONSISTENT` -- its 2 missing units are true at reset |
| `rearranging_kitchen_furniture` | – | – | – | – | **BROKEN -- no signal** |
| `storing_food` | – | – | – | – | **BROKEN -- no signal** |

Corpus-wide, only **4 of 98** measurable tasks have units the reward never credits in any
episode. The other 49 tasks that read `UNDER_FIRES` have sound instrumentation and
truncated demos.

`chop_an_onion` is the cleanest illustration. Its four units all register in 200 of 200
episodes; what varies is how far the demo gets before it ends. Grouping episodes by how
many units they reach, and asking when each level was first reached:

| units reached | episodes | t(1) | t(2) | t(3) | t(4) |
|---|---|---|---|---|---|
| 4 | 152 | 0.726 | 0.767 | 0.885 | **0.943** |
| 3 | 44 | 0.699 | 0.785 | **0.943** | – |
| 2 | 4 | 0.695 | **0.783** | – | – |

The early levels fire at the same times in all three groups, and whichever level is the
last one reached lands at t ~ 0.94 -- the episode ends on it. These are the same
trajectory cut at different depths. There is no predicate to drop and no `D = 3` version
of the task; there is a demo corpus in which 24% of episodes stop one step early, and
under the init anchor those episodes are labelled with the 0.75 they actually reached.

`wash_dog_toys` was the one genuinely systematic case, and it is not an instrument
defect either. It credits exactly 4 of D=6 in 196 of 199 episodes, with no variance. The
two units that never fire are exactly the two literals the census marks
`initially_true` -- `not covered(teddy_1, dust)` and `not covered(teddy_2, dirt)` -- which
start satisfied and are never broken, so they emit no delta. Anchored at init they are
`s0`, the task reads `CONSISTENT`, and its labels run from 0.333 to 1.0.

## Warning 3 -- a clean firing record does NOT prove complete instrumentation

`never_credited` measures firing relative to D. If `D` was collapsed below the true goal
size before measurement, `never_credited = 0` merely confirms `D` and the observed events
agree. It says nothing about whether `D` matches the BDDL goal, and no statistic derived
from the reward trace can see this.

Comparing measured `D` against BDDL predicates that must flip (`../census/`):
**31 tasks** have `D` < must-flip count and are listed in `task_shortlist.csv` under
`tier = excluded_collapsed_denominator`. `sorting_bottles_cans_and_paper` is the clearest:
16 predicates must flip, `D = 3`, and 3 does not even divide 16.

**A task is trustworthy only when `D` equals the must-flip count AND some demo reaches
progress 1.0.** `denominator_status` reports this as
`CONSISTENT` / `ENDS_SHORT` / `DEAD_UNITS` / `COLLAPSED`.

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

Seven tasks read `phi0 = 0` on their sample and do not on the full corpus.
`installing_a_fax_machine` was rank 9 of an earlier shortlist on the strength of
`phi0 = 0`. The sampling point stands on its own -- a sample flatters whatever statistic
you measure on it -- but note that these are terminal-anchored numbers, so what the full
corpus actually revealed on most of these seven was truncated demos rather than an
under-firing reward. `installing_a_fax_machine` is `ENDS_SHORT` with 157 of 200 episodes
completing; see warning 2.

**The null baseline, recomputed.** It was defined as the task-equal mean `phi0` over the
94 tasks whose instrumentation is `ok`, and it moved from 0.0932 (sample) to 0.1293 (full
corpus). Both numbers were terminal-anchored, so both were summing two unrelated things.
Init-anchored, over the same 94 tasks:

| quantity | task-equal mean |
|---|---|
| **progress already true at reset -- the real free credit** | **0.0466** |
| goal units never credited / D | 0.1106 |
| units earned then lost / D | 0.0040 |
| *old terminal-anchored `phi0` for comparison* | *0.1293* |

The three do not sum to 0.1293 exactly because the two anchors keep slightly different
episode sets. The shape of the answer is what matters: **roughly two thirds of the old
baseline was demos stopping short, not progress handed to the policy at reset.**

Two things follow for the success target.

1. **The label-side free credit is 0.0466, not 0.1293.** Only 19 of 94 tasks have any
   reset offset at all, and it concentrates: `can_meat` 0.552, `put_together_a_basic_pruning_kit`
   0.456, `wash_dog_toys` 0.333, `cook_a_brisket` 0.327.
2. **The eval-side null is 0.** This is the more important of the two. `TaskMetric`
   scores a goal literal that is already true at reset as zero forever (`../../docs/PREDICATE_CENSUS.md`),
   so a do-nothing policy scores `Q = 0` no matter how much of the goal starts satisfied.
   The old 0.1293 was never a score anything could earn; it was a property of the labels.
   Do not carry it into the A/B as a floor to beat.

## Warning 5 -- zero-headroom episodes

One `vacuuming_floors` episode and two `hanging_pictures` episodes **terminate having
fired no reward at all**. Terminal-anchored this read as `phi0 = 1` -- progress the
constant 1.0, a label teaching that a freshly-reset scene is already complete.
Init-anchored it reads as the constant 0.0, which is the truthful description. Either
way the label never changes and there is nothing to learn from it, so `labels.py` drops
them as `no_headroom`, now defined as *progress is constant for the whole episode*. They
are invisible in the sample.

## Warning 6 -- non-monotonic progress, and what it is made of

A genuine predicate flip-back survives repair by design, so progress can dip mid-episode.
The counts are large -- 197 of 197 valid `cook_bacon` episodes, 146 of 200
`chop_an_onion` -- and on their own they are not informative. `max_dip_units` is:

| Task | non-monotonic | deepest dip | what dips |
|---|---|---|---|
| `cook_bacon` | 197 / 197 | **1 of 7 units** | the refrigerator door, every time |
| `chop_an_onion` | 146 / 200 | 1 of 4 | knife and board in and out of the sink |
| `putting_shoes_on_rack` | 167 / 195 | 2 of 10 | shoes lifted off the rack again |

**`cook_bacon`'s dip is entirely `not open(electric_refrigerator)`.** Every valid episode
has exactly one of two signatures -- `(+1, +6)` in 153 and `(+1, -1, +1, +6)` in 44, all
in sevenths -- and every `±1` event precedes the single `+6`. The six `cooked` literals
never move until they all move at once, so no bacon predicate ever un-satisfies. Under
the old recipe 156 of those debits were repaired away and 44 survived, which is where the
"22% non-monotonic" figure came from; the split was an artefact of the repair floor
sitting at 0 rather than at `-s0/D`. All 200 episodes open the fridge.

`labels.py` does not smooth any of it -- the dips are real predicate state -- but a
progress head trained with a monotonicity prior will fight them, and on `cook_bacon` it
would be fighting a refrigerator door that earns nothing in `Q` (the literal is true at
reset, so `TaskMetric` scores it zero forever). Decide this before training, not after.

---

## Task selection

Budget is 2–8 tasks at ~64 GPU-hours each. Two things are being selected for, and they
are not the same thing:

**Trustworthy** -- the reward can express the whole goal and the labels are correct.
`denominator_status` and `demo_completion_rate`.

**Graded** -- the label carries a gradient a progress head can learn from.
`gradient_score = frac_intermediate * (1 - max_step_frac)`.

`D` is not a proxy for the second, and treating it as one selects for exactly the tasks
that cannot show whether shaping works. See the next section.

Criteria for the primary tier, in priority order:

1. **`denominator_status == CONSISTENT`** -- `D` equals the BDDL must-flip count and every
   valid demo reaches progress 1.0.
2. **Short mean episode length** -- eval timeout is `1.5x` the task's own mean demo
   length, so a short task is far cheaper to evaluate.
3. **`valid_rate >= 0.70`** and at least 30 episodes measured -- all 200 for every task
   below.

**Every qualifying task is listed. The tier is not truncated.** A fixed `head(12)` is how
`make_pizza` -- `CONSISTENT`, valid 0.995, 200 of 200 episodes complete -- came to be the
thirteenth of thirteen qualifying tasks and appear nowhere in a twelve-row shortlist while
this file claimed all thirteen were in the primary tier.

### Why length dominates

Full-corpus mean demo length runs **2,150 to 27,000+ frames** (corpus mean 10,546 over all
20,000 episodes). Eval cost is essentially linear in the timeout, so task choice alone
swings the eval bill by an order of magnitude. `rel_eval_cost` is normalised to the corpus
mean: the top pick costs **0.20x** an average task.

### Primary tier -- all 14 `CONSISTENT` tasks

| # | Task | D | Mean len | Rel cost | Valid | Offset at reset | Gradient | Largest step |
|---|---|---|---|---|---|---|---|---|
| 1 | `turning_on_radio` | 1 | 2,150 | 0.20x | 1.000 | 0.0 | 0.00 | 1.00 |
| 2 | `hanging_pictures` | 1 | 2,387 | 0.23x | 0.990 | 0.0 | 0.00 | 1.00 |
| 3 | `vacuuming_floors` | 1 | 2,412 | 0.23x | 0.995 | 0.0 | 0.00 | 1.00 |
| 4 | `installing_smoke_detectors` | 1 | 2,569 | 0.24x | 1.000 | 0.0 | 0.00 | 1.00 |
| 5 | `scrubbing_bathroom_floor` | 1 | 3,154 | 0.30x | 1.000 | 0.0 | 0.00 | 1.00 |
| 6 | `make_cabinet_doors` | 1 | 3,442 | 0.33x | 1.000 | 0.0 | 0.00 | 1.00 |
| 7 | `clean_a_keyboard` | 1 | 3,876 | 0.37x | 1.000 | 0.0 | 0.00 | 1.00 |
| 8 | `attach_a_camera_to_a_tripod` | 1 | 3,912 | 0.37x | 1.000 | 0.0 | 0.00 | 1.00 |
| 9 | `clean_a_trumpet` | 1 | 5,307 | 0.50x | 0.990 | 0.0 | 0.00 | 1.00 |
| 10 | `store_honey` | 1 | 6,767 | 0.64x | 1.000 | 0.0 | 0.00 | 1.00 |
| 11 | `cook_bacon` | 7 | 7,680 | 0.73x | 0.985 | **0.117** | 0.08 | **0.86** |
| 12 | `wash_dog_toys` | 6 | 11,223 | 1.06x | 0.995 | **0.333** | 0.00 | 0.33 |
| 13 | `clean_a_patio` | 1 | 12,071 | 1.14x | 1.000 | 0.0 | 0.00 | 1.00 |
| 14 | `make_pizza` | 2 | 19,187 | 1.82x | 0.995 | 0.0 | 0.00 | 0.50 |

Three of these joined or moved on re-measurement: `wash_dog_toys` was `UNDER_FIRES` and is
`CONSISTENT` with a 1/3 reset offset; `make_pizza` was invisible below the cut;
`cook_a_frozen_pie` left, having been listed here at rank 12 on a 28.4% sample -- across
all 200 episodes it is `ENDS_SHORT`, with 177 non-monotonic episodes and 1.1% of its
episodes stopping short of the goal.

### D is not gradient: why this tier cannot carry the experiment

**Twelve of the fourteen have `gradient_score = 0.00`.** Eleven are `D = 1`, where progress
is 0 until it is 1 and there is nothing to regress. The other three are worse than they
look:

- **`cook_bacon`, `D = 7`.** All 200 episodes contain exactly one `+6/7` reward sample, in a
  single frame, a median of 112 frames -- **1.5%** -- before the end, with nothing after it.
  Six bacon slices cross the cook threshold in the same physics step. Its label is 0 for
  35% of the episode, 1/7 for 56%, and 1.0 for the last 1.7%: a step function wearing a
  multi-unit denominator. `max_step_frac` 0.86.
- **`wash_dog_toys`, `D = 6`.** All four live units fire in the same frame at t = 0.948.
- **`make_pizza`, `D = 2`.** Both units fire together at t ~ 0.97; `frac_intermediate`
  0.0002, at 1.82x eval cost.

Corpus-wide, **22 of the 62 eligible tasks are effectively step functions**
(`max_step_frac >= 0.5`).

### Graded tier -- ranked by gradient, cost capped at 1.10x

| # | Task | D | Gradient | Frac intermediate | Largest step | Levels | First credit at | Demos completing | Rel cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `putting_shoes_on_rack` | 10 | **0.72** | 0.80 | 0.10 | 9.8 | t=0.15 | 33% | 0.73x |
| 2 | `outfit_a_basic_toolbox` | 7 | 0.69 | 0.81 | 0.14 | 7.0 | t=0.47 | 79% | 1.01x |
| 3 | `thawing_frozen_food` | 7 | 0.68 | 0.80 | 0.15 | 6.1 | t=0.14 | 9% | 0.78x |
| 4 | `set_up_a_coffee_station_in_your_kitchen` | 6 | 0.67 | 0.81 | 0.17 | 5.4 | t=0.32 | 32% | 0.59x |
| 5 | `preparing_lunch_box` | 6 | 0.60 | 0.72 | 0.17 | 5.6 | t=0.28 | 53% | 0.78x |
| 6 | `setting_the_fire` | 8 | 0.48 | 0.56 | 0.14 | 7.7 | t=0.42 | 74% | 0.86x |
| 7 | `put_together_a_basic_pruning_kit` | 4 | 0.47 | 0.63 | 0.25 | 3.8 | t=0.70 | 57% | 1.07x |
| 8 | `polishing_shoes` | 5 | 0.46 | 0.57 | 0.20 | 5.2 | t=0.26 | 44% | 1.02x |

All eight are `ENDS_SHORT`: sound instrumentation, demos that stop short. `putting_shoes_on_rack`
spends 80% of its episodes at intermediate progress across ten evenly spaced levels
(t = 0.14, 0.28, 0.38, 0.43, 0.60, 0.73, 0.88, 0.91, 0.95, 0.96) at 0.73x cost.

Read `demo_completion_rate` before picking. **`thawing_frozen_food` is listed but not
recommended:** at 9% completion a head sees almost no example of what finished looks like
(`--drop-incomplete-demos` would leave it 18 episodes), and it pairs that with
`census_alignment: ambiguous` and a repair on 198 of 200 episodes -- too much unquantified
risk for a task the experiment depends on. `outfit_a_basic_toolbox` at 79% is the safest
of the four on completion; `putting_shoes_on_rack` is the safest on the reset offset.

### What to check before training on a graded task

| Task | Literals true at reset | Unknown at reset | Offset | Repairs / 200 eps | Risk |
|---|---|---|---|---|---|
| `putting_shoes_on_rack` | 0 | 10 | 0.000 | **6** | low -- almost no trace behaves as if a literal was broken |
| `outfit_a_basic_toolbox` | 2 | 5 | 0.266 | 6 | low; offset is real and known (`ontop(toolbox, tabletop)`, `not open(toolbox)`) |
| `set_up_a_coffee_station` | 1 | 5 | 0.165 | 28 | moderate; offset is `ontop(coffee_maker, countertop)` |
| `thawing_frozen_food` | – | 6 | 0.000 | **198** | **high** -- `census_alignment: ambiguous` (9 literals, D=7) and a repair on nearly every episode. Check against a live reset first |

A high repair count with unknown initial states is the signature of an initially-true
literal being broken, its debit deleted and its restoring credit kept -- an over-credit of
up to `initially_true_unknown` units. `putting_shoes_on_rack` has ten unknowns and six
repairs in 200 episodes, so the exposure is empirically near zero. `thawing_frozen_food`
has six unknowns and 198 repairs.

### Suggested slate

The contribution needs at least one task where progress is not a step function, and the
primary tier does not contain one. **`putting_shoes_on_rack` (0.73x) plus
`set_up_a_coffee_station_in_your_kitchen` (0.59x)** carry the experiment; the cheap
`CONSISTENT` D=1 tasks -- `turning_on_radio` (0.20x), `installing_smoke_detectors` (0.24x),
`vacuuming_floors` (0.23x) -- are the control that the pipeline works end to end. Total
**1.99x** the eval cost of a single average task.

The one open question on `putting_shoes_on_rack` is observability, not labelling: its ten
literals are `touching` / `not touching` / `nextto`, which `../../docs/PREDICATE_CENSUS.md`
buckets as C (no discriminative visual signature) on a judgment call it flags as
debatable. `set_up_a_coffee_station` is all `ontop` / `nextto`, bucket A throughout, and is
the safer of the two on that axis.

### The organizers' two suggestions

- **`turning_on_radio` -- rank 1.** Confirmed excellent: shortest task in the corpus at
  2,150 frames, validity 1.000 across all 200 episodes, `D = 1` matching its single
  `toggled_on` literal, nothing true at reset, every demo completes. Take it as a control.
  It cannot show a shaping effect -- `gradient_score` 0.00.
- **`make_microwave_popcorn` -- excluded.** Short (3,238), and every episode completes. The
  audit says otherwise: its goal is **two** literals -- `real(cooked__popcorn)` and
  `contains(popcorn__bag, cooked__popcorn)` -- and `D = 1`. Both must flip; reward fires
  once. `COLLAPSED`. It would train (self-consistent, full headroom) but half its goal is
  invisible to the reward, so it can never demonstrate that shaping works.

---

## Instrumentation audit (task 3)

Method: read each task's BDDL goal literals from `../census/goal_census_detail.json`, count
those not already true at init (**must flip**), and compare against `D` and against what
the reward actually credits, init-anchored. Full table for all 98 tasks with a measurable
`D` in `bddl_audit.csv`.

| Verdict | Meaning | Tasks |
|---|---|---|
| `CONSISTENT` | `D` == must-flip count, and every valid demo reaches progress 1.0 | **14** |
| `ENDS_SHORT` | `D` right, instrumentation sound, some demos stop a unit or two short | **49** |
| `DEAD_UNITS` | not one demo in 200 ever shows the whole goal | **4** |
| `COLLAPSED` | `D` below the must-flip count | **31** |
| no reward signal at all | | **2** |

The four `DEAD_UNITS` tasks are `putting_dishes_away_after_cleaning` (12.99 of 14 units
never credited), `assembling_gift_baskets` (12.13 of 16), `sorting_vegetables` (8.90 of 13)
and `hiding_Easter_eggs` (3.40 of 9). Three of the four were on the earlier SUSPECTED list;
`canning_food`, the fourth name on it, is `ENDS_SHORT`.

The previous count of this table was 13 `CONSISTENT` and 54 `UNDER_FIRES`. The 54 have
resolved into 4 real instrumentation defects and 49 tasks whose demos truncate, plus
`wash_dog_toys` moving to `CONSISTENT`. **63 of 98 tasks are safe to build a progress
label from**, against 13 before.

---

## Files

| File | What it is |
|---|---|
| `labels.py` | **The label pipeline.** Parquet in, per-frame `progress` + `satisfied_count` out, with refusals. |
| `tests/test_labels.py` | 45 tests against real parquet, including the refusal cases, the emitted-schema pin and the corpus regression pin. |
| `measure_corpus.py` | Re-measures `D`, `phi0`, validity on **all 200** episodes of every locally-held task; also emits corpus-wide lengths. |
| `build_shortlist.py` | Regenerates `task_shortlist.csv` and `bddl_audit.csv`. |
| `og_state_decoder.py` | Decodes OmniGibson flat state vectors into named object poses/joints via the `scene_file` attr in each HDF5. Handles the assisted-grasp sentinel block and carry-forward of absent objects. |
| `behavior1k_reward_map.json` | Per task: `D`, `phi0_mean`, counts, validity, `reward_instrumentation`, `usable_for_labels`. The file `labels.py` loads. **Sample-scope.** |
| `behavior1k_task_table.csv` | 100 rows, per-task measurement detail. **Sample-scope.** |
| `behavior1k_episode_stats.csv` | 5,677 rows, one per episode. **Sample-scope.** The regression test pins against this. |
| `full_corpus_lengths.csv` | All 100 tasks, mean/median/p90 length over all 20,000 episodes, plus eval timeout and relative cost. |
| `full_corpus_remeasure.csv` | All 100 tasks re-measured over all 200 episodes, under **both** anchors: `phi0_mean` is terminal-anchored (what the shortlist used to be built on), `phi0_init_mean` / `never_credited_mean` / `lost_at_end_mean` / `demo_completion_rate` are init-anchored, plus the `progress_structure` columns. |
| `task_shortlist.csv` | Training-candidate ranking (task 2), with the audit columns, the reset-offset columns and `gradient_score`. Tiers: `primary` (every `CONSISTENT` task, by cost), `graded` (by gradient), `excluded_collapsed_denominator`. |
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
