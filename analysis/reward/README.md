# Reward-shaping artifacts for BEHAVIOR-1K

Everything here was measured on the LeRobot demo set on the Jetson: **5,677 episodes across
100 challenge tasks**. `D` is measured for **98** of them (2 tasks emit no reward signal at all).

These files are the input to reward-shaped labelling. Nothing here needs a GPU to reproduce.

---

## The recipe

For each task, from the demo rewards alone:

1. **Measure the denominator.** `D = 1 / min|reward|` over all non-zero reward samples in the task.
2. **Verify it.** Every observed reward magnitude must be an integer multiple of `1/D`.
   If any magnitude is not, `D` is wrong and the task is rejected.
3. **Build the potential.** With `cumsum_t` the running reward sum and `cumsum_final` its
   end-of-episode value:

   ```
   phi_t = D * (1 - cumsum_final + cumsum_t)
   ```

   `phi_t` counts **goal units still outstanding at time t**. It decreases by exactly 1
   each time a goal predicate flips, and reaches 0 at a successful terminal state.

### Validation

An episode is kept only if all of:

- **Terminal agreement** — the reward trace agrees with `next.terminated`.
- **Range** — `phi_0` in `[0, 1]` after normalising by `D`.
- **Integrality** — `phi_0 * D` is an integer.

Episodes failing any check are dropped. **~17% of episodes are dropped** corpus-wide
(4,723 of 5,677 valid = 83.2%). Integrality holds on **4,723 / 4,723** kept episodes — it
never fails once the first two checks pass.

### The identity that makes the audit possible

Because `phi_0 * D` is always an integer:

```
phi_0 * D        ==  goal units that NEVER fired during the demo
D * (1 - phi_0)  ==  reward events actually observed
```

A demo that reaches the goal should credit every unit, so **a clean task has `phi_0 = 0`**.
Any `phi_0 > 0` on a successful demo means reward under-fired. This is what caught
`putting_dishes_away_after_cleaning`.

---

## Files

| File | What it is |
|---|---|
| `og_state_decoder.py` | Decodes OmniGibson flat state vectors into named object poses/joints, using the `scene_file` attr in each HDF5. Handles the assisted-grasp sentinel block and carry-forward of absent objects. |
| `behavior1k_reward_map.json` | Per task: `D`, `phi0_mean`, episode counts, validity, `reward_instrumentation`, `usable_for_labels`. The file to load at training time. |
| `behavior1k_task_table.csv` | 100 rows, full per-task measurement detail (magnitudes, multiflip, repair rates, phi0 distribution). |
| `behavior1k_episode_stats.csv` | 5,677 rows, one per episode. |
| `lerobot_sweep_ep_stats.csv` | 316-episode sweep with each validation gate broken out (`term_ok`, `inrange`, `integral`). |
| `reward_denominators.json` | The 12 tasks used to derive the recipe: observed `D` vs top-level conjuncts vs quantifier-expanded count. |
| `task_goal_terms.json` | Top-level BDDL conjunct counts. **Not a source of `D`** — see below. |
| `task_shortlist.csv` | Training-candidate ranking (task 2). |
| `build_shortlist.py` | Regenerates `task_shortlist.csv` and the audit columns. |

---

## Warning 1 — `task_goal_terms.json` is not a source of `D`

It counts **top-level BDDL conjuncts**, which is not what the reward divides by. It agrees
with the measured `D` on only **67 of 98** tasks. The failure is quantifiers: `picking_up_trash`
has 1 top-level conjunct (`forall` over three cans) but the reward uses `D = 3`.

**Always measure `D` from the reward trace. Retained for reference only.**

---

## Warning 2 — the instrumentation defect

On some tasks the reward under-fires: the demo reaches the goal but reward never credits
every unit. **Validation does not catch this** — such episodes pass all three gates, because
the gates check internal consistency, not whether the reward matches the BDDL goal.

The signature is `phi_0`, which counts units that never fired:

- **`phi_0 = 0`** — every unit fired. Good instrumentation.
- **high `phi_0`** — suspect. Most of the goal was never credited.

### Named tasks

| Task | D | phi0 | Reward events (of D) | Status |
|---|---|---|---|---|
| `putting_dishes_away_after_cleaning` | 14 | 0.929 | 1.0 of 14 | **PROVEN incomplete** |
| `sorting_vegetables` | 13 | 0.703 | 3.9 of 13 | SUSPECTED |
| `assembling_gift_baskets` | 16 | 0.699 | 4.8 of 16 | SUSPECTED |
| `canning_food` | 10 | 0.638 | 3.6 of 10 | SUSPECTED |
| `rearranging_kitchen_furniture` | – | – | none | **BROKEN — no signal** |
| `storing_food` | – | – | none | **BROKEN — no signal** |

**`putting_dishes_away_after_cleaning` is the proven case**: `D = 14`, but a successful demo
fires reward **exactly once**. Thirteen of fourteen goal units are never credited. Its BDDL goal
is 8 `inside(plate, cabinet)` literals plus 2 `not open(cabinet)` literals, and the denominator
is scene-dependent (`cabinet.n.01_*` wildcard), so `D` varies. `rearranging_kitchen_furniture`
and `storing_food` emit **no reward at all** (validity 0.0) and have no measurable `D`.

---

## Warning 3 — `phi_0 = 0` does NOT prove complete instrumentation (new)

Found while running the task-3 audit. This **corrects the rule stated above** and it changes
which tasks are safe.

`phi_0` measures firing *relative to D*. If `D` itself was collapsed below the true goal size
**before** measurement, `phi_0 = 0` merely confirms `D` and the observed events agree — it says
nothing about whether `D` matches the BDDL goal. This is a second, independent defect mode,
and `phi_0` is blind to it.

Comparing measured `D` against BDDL predicates that must flip (`analysis/census/`), on
scene-invariant tasks only:

- **32 tasks** have `D` < predicates that must flip.
- **18 of them have `phi_0 = 0` in every episode and are marked `ok`** — completely invisible
  to the existing validation.
- That is **18 of the 41 `phi_0 = 0` tasks (44%)**.

Worst offenders, all `phi_0 = 0` and all previously marked `ok`:

| Task | D | Must flip | Hidden units |
|---|---|---|---|
| `putting_away_toys` | 1 | 8 | 7 |
| `laying_tile_floors` | 2 | 8 | 6 |
| `stacking_wood` | 1 | 6 | 5 |
| `collecting_aluminum_cans` | 1 | 6 | 5 |
| `packing_meal_for_delivery` | 2 | 6 | 4 |
| `turning_out_all_lights_before_sleep` | 2 | 5 | 3 |
| `dispose_of_glass` | 1 | 4 | 3 |
| `store_produce` | 1 | 4 | 3 |

`sorting_bottles_cans_and_paper` is the clearest: 16 predicates must flip, `D = 3`, and 3 does
not even divide 16. `putting_away_toys` and `stacking_wood` are `D = 1` binary rewards standing
in for 8- and 6-predicate goals.

**Consequence:** a task is only trustworthy when `phi_0 = 0` **and** `D` equals the number of
BDDL predicates that must flip. `task_shortlist.csv` reports this as `denominator_status`
(`CONSISTENT` / `COLLAPSED` / `UNDER_FIRES`).

---

## Task selection (task 2)

Budget is 2–8 tasks at ~64 GPU-hours each. Criteria, in the stated priority order:

1. **`phi_0 = 0` in every episode** — full headroom and (necessary, not sufficient — see
   warning 3) evidence of complete instrumentation. **41 of 98** tasks qualify.
2. **Short mean episode length** — the eval timeout is `1.5 x` the task's own mean demo
   length, so a short task is far cheaper to evaluate. **This is the criterion that had not
   been applied before, and it reshuffles the list completely.**
3. **Validity rate >= 70%** — drops 41 to **34**. (Redundant with `usable_for_labels`, which
   already gates on exactly this threshold.)
4. **Enough episodes** — every task below has >= 39 valid episodes.

### Why length dominates

Mean demo length across the corpus runs **2,312 to 27,584 frames — a 12x spread** (corpus
mean 10,842). Eval cost is essentially linear in the timeout, so task choice alone swings the
eval bill by an order of magnitude. `rel_eval_cost` in the shortlist is normalised to the
corpus mean: **the top pick costs 0.21x an average task; `boxing_books_up_for_storage` would
cost 2.5x.** Picking the eight shortest `phi_0 = 0` tasks instead of eight average ones cuts
the eval budget by roughly **70%**.

### Ranked shortlist

Full detail in `task_shortlist.csv`.

| # | Task | D | Mean len | Timeout | Rel cost | Valid | Eps | Denominator |
|---|---|---|---|---|---|---|---|---|
| 1 | `vacuuming_floors` | 1 | 2312 | 3468 | 0.21x | 1.0 | 105 | CONSISTENT |
| 2 | `turning_on_radio` | 1 | 2342 | 3514 | 0.22x | 1.0 | 98 | CONSISTENT |
| 3 | `hanging_pictures` | 1 | 2407 | 3611 | 0.22x | 1.0 | 94 | CONSISTENT |
| 4 | `installing_smoke_detectors` | 1 | 2518 | 3776 | 0.23x | 1.0 | 96 | CONSISTENT |
| 5 | `scrubbing_bathroom_floor` | 1 | 3113 | 4670 | 0.29x | 1.0 | 77 | CONSISTENT |
| 6 | `make_microwave_popcorn` | 1 | 3374 | 5061 | 0.31x | 1.0 | 68 | COLLAPSED |
| 7 | `make_cabinet_doors` | 1 | 3535 | 5303 | 0.33x | 1.0 | 68 | CONSISTENT |
| 8 | `clean_a_keyboard` | 1 | 3876 | 5814 | 0.36x | 1.0 | 62 | CONSISTENT |
| 9 | `installing_a_fax_machine` | 2 | 3914 | 5871 | 0.36x | 0.71 | 62 | CONSISTENT |
| 10 | `attach_a_camera_to_a_tripod` | 1 | 3987 | 5980 | 0.37x | 1.0 | 57 | CONSISTENT |
| 11 | `sweeping_garage` | 2 | 4475 | 6713 | 0.41x | 0.963 | 54 | CONSISTENT |
| 12 | `clean_a_trumpet` | 1 | 5695 | 8542 | 0.53x | 0.975 | 40 | CONSISTENT |

**Recommended 4 if the budget is tight:** `vacuuming_floors`, `turning_on_radio`,
`hanging_pictures`, `installing_smoke_detectors` — all `D = 1`, `phi_0 = 0`, validity 1.00,
`CONSISTENT` denominators, 94–105 episodes each, and all under 0.25x eval cost.
They are also varied in skill: surface cleaning, a toggle, and two `attached()` placements.

### The organizers' two suggestions

- **`turning_on_radio` — rank 2.** Confirmed excellent. Shortest but one, validity 1.00,
  98 episodes, `D = 1` matching its single `toggled_on` literal. Take it.
- **`make_microwave_popcorn` — rank 6 on cost, but flagged `COLLAPSED`.** It is short (3,374)
  and `phi_0 = 0` with validity 1.00, so on the stated criteria it looks clean. The audit says
  otherwise: its goal is **two** literals — `real(cooked__popcorn)` and
  `contains(popcorn__bag, cooked__popcorn)` — and `D = 1`. Both must flip; reward fires once.
  It is safe to train on (self-consistent, full headroom) but it carries **zero** progress
  structure and one of its two subgoals is invisible to the reward. **Usable, not a
  demonstration of reward shaping.**

### Medium-D candidates (deliberate)

The `phi_0 = 0` set skews hard to `D = 1–2` — of the 34 qualifying tasks, **23 are `D = 1`**
and only **one** has `D >= 4`. A reward that is nearly binary gives a shaping contribution
almost nothing to learn from, so we should carry at least one task with real progress
structure even at higher cost:

| Task | D | Mean len | Rel cost | Valid | phi0 | Denominator | Why |
|---|---|---|---|---|---|---|---|
| `cook_bacon` | 7 | 8547 | 0.79x | 0.962 | 0.0 | CONSISTENT | **No tradeoff.** The only D>=4 task with phi0=0. 6 bacon slices cook independently — 7 clean units. |
| `make_rose_centerpieces` | 4 | 4402 | 0.41x | 0.889 | 0.0938 | UNDER_FIRES | Shortest medium-D task (0.41x). 3 roses + vase placement. Mild under-fire. |
| `chop_an_onion` | 4 | 6385 | 0.59x | 0.943 | 0.0568 | UNDER_FIRES | 70 episodes, validity 0.94. Mixed predicate types (slice, place, contain). |

**The tradeoff, stated explicitly.** Criterion 1 and the need for progress structure pull in
opposite directions: `phi_0 = 0` selects for goals so simple they cannot under-fire, which is
exactly the set with no shaping signal. Medium-D tasks cost **2–4x more per eval** and mostly
carry small non-zero `phi_0`.

**`cook_bacon` resolves it.** `D = 7`, `phi_0 = 0` exactly, validity 0.96, and all 7 units fire.
It is the one task in the corpus that is both structurally rich and provably clean. One caveat:
one of its 7 literals (`not open(electric_refrigerator)`) is already true at init, so 6 units
represent real work. At 0.79x cost it should be in any shortlist of 4 or more.

**Suggested 6-task slate:** the recommended 4, plus `cook_bacon` (structure) and
`make_rose_centerpieces` (structure at low cost). Total ~2.0x the eval cost of a *single*
average task.

---

## Instrumentation audit (task 3)

Method: read each task's BDDL goal literals from `analysis/census/goal_census_detail.json`,
count those not already true at init (**must flip**), and compare against reward events
actually observed, `D * (1 - phi_0)`. Literals with unknown init state are counted as needing
to flip.

| Task | D | Must flip | Literals | Reward events | Verdict |
|---|---|---|---|---|---|
| `vacuuming_floors` | 1 | 1 | 1 | 1.0 | **clean** |
| `turning_on_radio` | 1 | 1 | 1 | 1.0 | **clean** |
| `hanging_pictures` | 1 | 1 | 1 | 1.0 | **clean** |
| `installing_smoke_detectors` | 1 | 1 | 1 | 1.0 | **clean** |
| `scrubbing_bathroom_floor` | 1 | 1 | 1 | 1.0 | **clean** |
| `make_microwave_popcorn` | 1 | 2 | 2 | 1.0 | **suspect** — D below goal size |
| `make_cabinet_doors` | 1 | 1 | 1 | 1.0 | **clean** |
| `clean_a_keyboard` | 1 | 1 | 1 | 1.0 | **clean** |
| `installing_a_fax_machine` | 2 | 2 | 2 | 2.0 | **clean** |
| `attach_a_camera_to_a_tripod` | 1 | 1 | 1 | 1.0 | **clean** |
| `sweeping_garage` | 2 | 2 | 2 | 2.0 | **clean** |
| `clean_a_trumpet` | 1 | 1 | 1 | 1.0 | **clean** |
| `cook_bacon` | 7 | 6 | 7 | 7.0 | **clean** (1 literal already true at init) |
| `make_rose_centerpieces` | 4 | 4 | 4 | 3.62 | **suspect** — units uncredited |
| `chop_an_onion` | 4 | 4 | 4 | 3.77 | **suspect** — units uncredited |
| `putting_dishes_away_after_cleaning` | 14 | 8 | 10 | 1.0 | **do not use** — reference defect |

**Result: 11 of the top 12 are clean.** The single flag is `make_microwave_popcorn`
(`COLLAPSED`, discussed above). The two medium-D picks `make_rose_centerpieces` and
`chop_an_onion` show small `UNDER_FIRES` — their denominators match the goal, but a minority
of demos end without crediting the last unit (~0.3 of 4 on average). That is ordinary demo
noise, not the `putting_dishes_away` failure mode; the dropped episodes are already excluded
by validation. Both remain usable.

---

## The 76 `usable_for_labels` tasks

The gate in `behavior1k_reward_map.json` is exactly:

```
usable_for_labels  ==  reward_instrumentation == "ok"  AND  valid_rate >= 0.70
```

The separation is clean: the lowest-validity included task is `installing_a_fax_machine` at
**0.7097**, the highest excluded is `clearing_food_from_table_into_fridge` at **0.6957**.

**It does not yet incorporate warning 3** — 18 of these 76 have collapsed denominators.
They are safe to *train* on but should not be cited as evidence that shaping works.

|  |  |  |  |
|---|---|---|---|
| `attach_a_camera_to_a_tripod` | `boxing_books_up_for_storage` | `bringing_in_wood` | `bringing_paper_to_recycling` |
| `bringing_water` | `can_meat` | `carrying_in_groceries` | `chop_an_onion` |
| `chopping_wood` | `clean_a_keyboard` | `clean_a_patio` | `clean_a_trumpet` |
| `clean_boxing_gloves` | `clean_up_broken_glass` | `clean_up_your_desk` | `clean_your_rusty_garden_tools` |
| `cleaning_up_branches_and_twigs` | `cleaning_up_plates_and_food` | `collecting_aluminum_cans` | `cook_a_brisket` |
| `cook_a_frozen_pie` | `cook_bacon` | `cook_broccolini` | `cook_brussels_sprouts` |
| `cook_cabbage` | `cook_hot_dogs` | `dispose_of_glass` | `freeze_fruit` |
| `freeze_pies` | `getting_organized_for_work` | `halve_an_egg` | `hanging_pictures` |
| `hiding_Easter_eggs` | `installing_a_fax_machine` | `installing_a_modem` | `installing_smoke_detectors` |
| `loading_the_car` | `make_cabinet_doors` | `make_microwave_popcorn` | `make_pizza` |
| `make_rose_centerpieces` | `organizing_art_supplies` | `outfit_a_basic_toolbox` | `picking_up_toys` |
| `picking_up_trash` | `polishing_shoes` | `preparing_lunch_box` | `put_together_a_basic_pruning_kit` |
| `putting_away_Halloween_decorations` | `putting_away_toys` | `putting_shoes_on_rack` | `putting_up_Christmas_decorations_inside` |
| `re_shelving_library_books` | `rearrange_your_room` | `scrubbing_bathroom_floor` | `set_up_a_coffee_station_in_your_kitchen` |
| `setting_mousetraps` | `setting_the_fire` | `slicing_vegetables` | `sorting_bottles_cans_and_paper` |
| `sorting_household_items` | `spraying_for_bugs` | `spraying_fruit_trees` | `stacking_wood` |
| `store_batteries` | `store_honey` | `store_produce` | `sweeping_garage` |
| `thawing_frozen_food` | `tidying_bathroom` | `turning_on_radio` | `turning_out_all_lights_before_sleep` |
| `unloading_the_car` | `vacuuming_floors` | `wash_a_baseball_cap` | `wash_dog_toys` |

**Excluded (24):** the 6 named in warning 2, plus **18 with `valid_rate < 0.70`** — the worst
being `tidying_bedroom` (0.05, only 2 valid episodes of 40), `dispose_of_batteries` (0.26) and
`tidying_living_room` (0.39). **No task fails magnitude integrality** — every task with a
measurable `D` has all magnitudes as exact multiples of `1/D`, so step 2 of the recipe has
never once rejected a task. It rules out a wrong `D`, not a bad task.
