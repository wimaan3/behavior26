# A/B/C predicate census — all 100 challenge tasks

Reproduce with:

```bash
python scripts/fetch_challenge_tasks.py --out analysis/census/challenge_tasks.json
python -m analysis.goal_census --bddl <BEHAVIOR-1K>/bddl3 \
    --tasks analysis/census/challenge_tasks.json --out-dir analysis/census
```

Outputs: `analysis/census/goal_census.csv` (per task),
`analysis/census/goal_census_detail.json` (per literal, with the reason for
each bucket assignment).

Sources: `StanfordVL/BEHAVIOR-1K` at tag **v3.9.2**, `bddl3` subpackage.
Verified 2026-08-27 that v3.9.2 is still the pinned tag on
behavior.stanford.edu/challenge/evaluation.html.

---

## How Q is actually computed — read this first

From `OmniGibson/omnigibson/metrics/task_metric.py`:

```python
if env.task.success:
    final_q_score = 1.0
else:
    final_q_score = max(
        sum(int(not initially_true and pred.evaluate(...))
            for pred, initially_true in zip(option, option_previous_state))
        / len(option)
        for option, option_previous_state in zip(
            env.task.ground_goal_state_options, self.initial_predicate_states)
    )
```

Three things follow, and they matter more than the census itself:

1. **The denominator is grounded literals, not surface conjuncts.**
   `picking_up_trash` has one goal conjunct — `(forall (?can) (inside ?can ?ashcan))` —
   but three declared cans, so the denominator is 3 and each can is worth 1/3.

2. **A goal literal that is already true at reset scores zero forever.**
   It stays in the denominator. `initial_predicate_states` is captured in
   `TaskMetric.reset()`, so re-achieving it after breaking it does not help.
   Only full `task.success` bypasses this by short-circuiting to 1.0.

3. **Q is a max over ground options.** Disjunctions and `exists` produce many
   options; the policy is scored against whichever it did best on. All 100
   tasks have options of uniform size, so the denominator is unambiguous.

---

## Bucket definitions

| | meaning |
|---|---|
| **A** | resolvable from the final RGB-D frame alone — geometry or appearance, normally in view |
| **B** | occludable: a real visual signature exists, but the scored thing can be hidden at episode end (inside a closed container, under furniture). Needs memory or active looking |
| **C** | no discriminative visual signature even with a perfect view |

## Headline result

| | count | share |
|---|---|---|
| grounded goal literals across 100 tasks | 550 | |
| **A** | 401 | **72.9%** |
| **B** | 122 | 22.2% |
| **C** | 27 | 4.9% |

Literals per task: mean 5.50, median 5, range 1–26.

- **59/100 tasks are entirely A.** A memory-free predicate head can in
  principle track all scored progress on them.
- **12/100 tasks have zero A literals.** A memory-free head is blind to all
  scored progress: `storing_food`, `hanging_pictures`,
  `attach_a_camera_to_a_tripod`, `make_microwave_popcorn`, `cook_cabbage`,
  `cook_hot_dogs`, `stacking_wood`, `installing_smoke_detectors`,
  `make_cabinet_doors`, `store_batteries`, `store_honey`, `store_produce`.
- **11 tasks are all-or-nothing** (denominator 1): `turning_on_radio`,
  `hanging_pictures`, `attach_a_camera_to_a_tripod`, `clean_a_patio`,
  `clean_a_trumpet`, `scrubbing_bathroom_floor`, `installing_smoke_detectors`,
  `vacuuming_floors`, `make_cabinet_doors`, `store_honey`, `clean_a_keyboard`.
  No partial credit available on any of them.

### Does the architecture need memory?

**Not for the first version.** Averaged over the 100 tasks, 65.2% of each
task's goal state is both A-bucket *and* not already satisfied at reset. A
purely reactive, memory-free predicate head has a Q ceiling of **0.652** — far
above any target on the table. Memory buys the last third and should be a
later experiment, not a week-2 dependency.

The concentration matters though: the B and C mass is not spread evenly. It
piles up in the 12 blind tasks and in one predicate, `inside` a closed-able
container (77 literals). If those 12 tasks are cheap to reach otherwise, they
are the natural place to spend a memory mechanism later.

## Per-predicate breakdown

| count | predicate | buckets |
|---|---|---|
| 232 | `inside` | A 155 / B 77 |
| 103 | `ontop` | A 103 |
| 41 | `nextto` | A 41 |
| 31 | `not open` | A 31 |
| 28 | `real` | A 19 / C 9 |
| 23 | `cooked` | B 23 |
| 21 | `not covered` | A 21 |
| 13 | `not inside` | A 13 |
| 10 | `touching` | C 10 |
| 6 | `not toggled_on` | A 6 |
| 5 | `toggled_on` | A 5 |
| 5 | `under` | B 5 |
| 5 | `not real` | B 5 |
| 4 | `not touching` | C 4 |
| 4 | `attached` | C 4 |
| 4 | `covered` | A 4 |
| 4 | `contains` | B 4 |
| 3 | `on_fire` | A 3 |
| 2 each | `frozen`, `filled`, `not contains`, `not frozen` | B |

### Evidence for the non-obvious calls

- **`toggled_on` → A.** `object_states/toggle.py` renders a visual marker
  coloured green when on and red when off (line 127) and sets
  `visible = True` unconditionally (line 171). This is the single most
  legible predicate in the whole set.
- **`inside` split by container.** `inside(x, b)` is A when `b` is not
  `openable` (bowl, ashcan, basket) and B when it is (fridge, cabinet,
  hinged jar, oven). Openability comes from
  `bddl/generated_data/propagated_annots_canonical.json`.
- **`cooked` / `frozen` → B.** Both define `get_texture_change_params`
  (brown tint, white tint), so a signature exists — but the object is
  normally inside an oven, microwave, pot or freezer at episode end.
- **`real(x)` split by substance.** From
  `bddl_utils.evaluate_bddl_predicate`, `real(x)` is simply "x exists" — the
  product of a transition rule. A new rigid object appearing (`half__log`) is
  A; a substance appearing inside a container (`cooked__popcorn` in a bag) is
  C. `not real(x)` (object consumed) is B: an occluded object looks the same
  as a destroyed one.
- **`touching` / `attached` → C.** These are PhysX contact and `AttachedTo`
  joint facts. Contact versus a 2 mm gap is not resolvable at 720×720, and an
  attached poster looks identical to one merely resting against the nail.
  **Judgment call** — they are geometric, not invisible. Reclassifying all 18
  of them as A takes A from 72.9% to 76.2% and empties C down to the 9
  substance-`real` literals. The table lives in
  `analysis/goal_census.py::UNCONDITIONAL` if we want to revisit.

## Initially-true literals — dead weight in the denominator

Estimated from the BDDL `:init` block under closed-world assumptions
(**not yet verified against a live reset** — see caveats):

- **57 of 550 literals (10.4%)** are already satisfied at reset, across
  **32 tasks**. They cannot contribute to partial credit.
- **29 of the 31 `not open` literals** are among them. Cabinets start closed.
  "Open the cabinet, load it, close it again" earns nothing for the closing —
  the literal was true at reset. It is still required for `success`.
- Worst-affected tasks: `canning_food` (6 of 10 dead, ceiling 0.40),
  `can_meat` (5 of 9, ceiling 0.44), `cleaning_up_plates_and_food` (4 of 8,
  ceiling 0.50), `put_together_a_basic_pruning_kit` (2 of 4, ceiling 0.50).
- No task has *every* literal initially true, so partial credit is available
  everywhere in principle.

## What this says about strategy

Satisfying **exactly one** not-initially-true goal literal in every one of the
100 tasks scores **Q = 0.321** — above the 2025 winning score of 0.260.
Fully solving 15 tasks and scoring zero on the other 85 gives Q = 0.150.
The breadth-first read of the metric is correct, and the margin is large.

Restricting that same one-literal-per-task strategy to only the 59 all-A tasks
gives Q = 0.198, still comfortably past the 0.10 target.

## Caveats

1. **The initially-true numbers are an estimate.** They assume unary states
   (`open`, `toggled_on`, `cooked`, …) default to false unless `:init` says
   otherwise, and they return "unknown" for kinematic relations not stated in
   `:init`. The scorer evaluates these in the live simulator at reset.
   Verify against a real rollout before trusting any single task's ceiling.
   The `not open` finding is the one most worth confirming first, since it is
   29 literals resting entirely on "cabinets load closed".
2. **Three tasks have scene-dependent denominators.**
   `putting_away_Halloween_decorations`, `cleaning_up_plates_and_food` and
   `putting_dishes_away_after_cleaning` quantify over wildcarded synsets
   (`cabinet.n.01_*`, `table.n.02_*`), which `bddl/wildcard.py` expands against
   the actual scene. Their counts here are the minimum; the real denominator
   for a given instance can be larger. 16 tasks contain wildcards; only these
   3 change size.
3. **A/B/C is about what a predicate head can *observe*, not about task
   difficulty.** An A-bucket literal can still be hard to achieve.
4. The bucket assignments are per-predicate plus argument type, not
   per-scene. A bowl on a high shelf is still classed A.
