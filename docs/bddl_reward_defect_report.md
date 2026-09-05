# BEHAVIOR-1K demo rewards: tasks where the reward stream cannot express the BDDL goal

**Status:** draft for Imaan. Not filed anywhere.
**Requested by:** organizer "stef", July 2026, who asked for a reproduction and did not get one.
**Measured on:** `behavior-1k/2026-challenge-demos` at `4f50b447` (2026-08-05), all 20,000
episodes of all 100 tasks. No simulator involved.

## Why this is a reproduction without a replay

The ask was for an hdf5 that reproduces the defect. We cannot replay demos (no Isaac Sim
here), but a replay is a weaker instrument than what the released demos already contain.
Each demo is a **successful** trajectory: it terminates, and it is the organizers' own
recording of the goal being achieved. So for every task we can compare two numbers that
should agree and do not:

- **`m`** — the number of BDDL goal literals that must flip over the episode, read from
  the task's goal definition (literals already true at initial state excluded).
- **`k`** — the number of goal units the reward stream actually credits over the episode,
  read from `next.reward`.

`k < m` on a demo that terminates successfully means the reward under-fires: the goal was
reached, and the reward never said so. That is a property of **every** episode of the
task, not a single trace, which makes it stronger evidence than one hdf5.

## Method

1. `D = 1 / min|next.reward|` over all non-zero reward samples of the task. Every
   observed magnitude is then checked to be an integer multiple of `1/D`; if any is not,
   `D` is wrong and the task is not reported here.
2. Rollback repair. The demo collector rolls the simulator back and replays, and the
   reward function then emits a debit for credit it never issued in this trace. Any
   negative that would drive the running sum below zero is zeroed — no real predicate
   flip can un-satisfy a unit that was never satisfied. A negative that leaves the sum at
   or above zero is a genuine flip-back and is kept.
3. `k = D * sum(repaired rewards)` — goal units credited over the episode.
4. `m` from the BDDL goal literals.

Two distinct defects fall out, and they need different fixes:

| Defect | Signature | Consequence |
|---|---|---|
| **UNDER_FIRES** | `D` is right, but `k < D` on a successful demo | reward omits credit it should have issued |
| **COLLAPSED** | `D < m` — the reward's own unit count is below the goal size | part of the goal is invisible to the reward at any `k` |

`COLLAPSED` is the more dangerous of the two because it is **self-consistent**: such a
task shows `k == D` and looks perfectly clean on every internal check. It can only be
caught by reading the BDDL.

## Case 1 — `putting_dishes_away_after_cleaning` (the clearest under-fire)

BDDL goal: 8 × `inside(plate.n.04_i, cabinet.n.01_1)` plus 2 `not open(cabinet)` literals
already true at init. **`m = 8`.** Measured **`D = 14`** (sole reward magnitude
`0.071429 = 1/14`; integrality passes).

Across **all 200 episodes**, every one of which terminates (`next.terminated` true on the
last frame, `next.truncated` never set):

| Goal units credited, `k` | Episodes |
|---|---|
| 1 | 198 |
| 2 | 1 |
| 3 | 1 |

Raw event counts per episode: 1–4 positive rewards and 1–3 negative rewards; 192 of the
negatives are rollback debits that would drive the running sum below zero. **Net credit is
1 unit of 14 in 198 of 200 episodes.**

So: 8 plates are placed in a cabinet, the demo terminates successfully, and the reward
credits one unit. `phi0 = 13/14 = 0.9286` — a progress label built from this stream would
declare a freshly reset scene 93% complete.

## Case 2 — tasks that emit no reward at all

`rearranging_kitchen_furniture`: **200 of 200 episodes terminate, and `next.reward` is
identically zero in every frame of every episode.** There is no reward magnitude, so `D`
is not even measurable. Its BDDL goal is 3 `inside(..., cabinet.n.01_1)` literals plus one
already-true `not open(cabinet)`, so `m = 3` and `k = 0`.

## Case 3 — `make_microwave_popcorn` (collapsed, and it looks clean)

BDDL goal is two literals, **both false at init**: `real(cooked__popcorn.n.01_1)` and
`contains(popcorn__bag.n.01_1, cooked__popcorn.n.01_1)`. **`m = 2`.**

Measured across all 200 episodes: the only reward magnitude is `1.0`, so **`D = 1`**. Every
episode fires exactly one positive reward and terminates. `k = 1 = D`, `phi0 = 0`,
validity 1.000.

This task passes every internal consistency check that exists. It is nonetheless wrong:
the reward has one unit to spend on a two-predicate goal, so **half the goal is invisible
to the reward**, and no amount of episode data reveals it. This is the case that most
needs an organizer fix, and the one most likely to be missed — it was recommended to
entrants as a good task.

## Corpus-wide result

Measured over **all 20,000 episodes of all 100 tasks** (no sampling). Of 100 tasks:

| Verdict | Tasks |
|---|---|
| **CONSISTENT** — `D` equals the must-flip count *and* every episode credits all `D` units | **13** |
| **UNDER_FIRES** — `D` is right, but successful demos credit fewer than `D` units | **54** |
| **COLLAPSED** — `D` is below the number of BDDL predicates that must flip | **31** |
| **no reward signal at all** | **2** |

So **87 of 100 tasks** have a reward stream that cannot be reconciled with their own BDDL
goal, and only 13 are safe to build a progress label from.

The 2 with no signal are `rearranging_kitchen_furniture` and `storing_food`: `next.reward`
is identically zero across 200 of 200 episodes, all of which terminate.

### Worst under-firing tasks

| Task | `D` | must flip `m` | reward events `k` | `phi0` |
|---|---|---|---|---|
| `putting_dishes_away_after_cleaning` | 14 | 8 | 1.02 | 0.927 |
| `assembling_gift_baskets` | 16 | 16 | 3.88 | 0.758 |
| `sorting_vegetables` | 13 | 13 | 4.10 | 0.685 |
| `canning_food` | 10 | 4 | 3.51 | 0.649 |
| `can_meat` | 9 | 4 | 4.88 | 0.458 |
| `getting_organized_for_work` | 10 | 7 | 5.96 | 0.404 |
| `hiding_Easter_eggs` | 9 | 9 | 5.60 | 0.378 |
| `tidying_bedroom` | 3 | 3 | 1.94 | 0.355 |
| `clearing_food_from_table_into_fridge` | 5 | 4 | 3.24 | 0.352 |
| `wash_dog_toys` | 6 | 4 | 4.00 | 0.333 |
| `organizing_school_stuff` | 6 | 6 | 4.11 | 0.315 |
| `put_together_a_basic_pruning_kit` | 4 | 2 | 2.78 | 0.304 |

### Collapsed denominators (these look clean on every internal check)

| Task | `D` | must flip `m` | `phi0` |
|---|---|---|---|
| `cook_brussels_sprouts` | 3 | 24 | 0.265 |
| `setup_a_bar_for_a_cocktail_party` | 9 | 20 | 0.277 |
| `sorting_bottles_cans_and_paper` | 3 | 16 | 0.049 |
| `cook_broccolini` | 3 | 11 | 0.003 |
| `setting_the_table` | 4 | 8 | 0.201 |
| `laying_tile_floors` | 2 | 8 | 0.055 |
| `putting_away_toys` | 1 | 8 | 0.000 |
| `chopping_wood` | 4 | 8 | 0.072 |
| `collecting_childrens_toys` | 6 | 7 | 0.067 |
| `sorting_books_on_shelf` | 6 | 7 | 0.414 |

### A sampling note the organizers may also want

`D = 1/min|reward|`, so it is set by the single smallest reward magnitude anywhere in the
task. On `wash_dog_toys` the magnitude `1/6` occurs **exactly once in 200 episodes**. Any
analysis that reads a subset of episodes will measure `D = 3` and be wrong by a factor of
two. This is a property of the data, not of our tooling, and it means per-task reward
statistics computed on anything less than the full episode set cannot be trusted.


## What would fix it

For the under-firing tasks the reward function is not crediting every satisfied predicate
— most visibly where a quantified goal (`forall` over 8 plates) collapses to a single
credit. For `COLLAPSED` tasks the denominator itself is wrong: `D` should equal the number
of goal literals that must flip, and where it does not, the reward cannot represent the
goal it is scoring.

A useful check the organizers could run in CI, needing no simulator: for every task, assert
`D == m` and assert that a successful demo credits `k == D`. Both numbers are already in
the released parquet and the BDDL.
