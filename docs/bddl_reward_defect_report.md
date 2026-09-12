# BEHAVIOR-1K demo rewards: 37 tasks where the reward stream cannot express the BDDL goal

**Status:** draft for Imaan. Not filed anywhere.
**Revised 2026-09-05** after an error in our own audit method. An earlier draft of this
report claimed 87 of 100 tasks were defective. That number was wrong and is retracted
below; the corrected claim is 37, of which the under-firing class is **4 tasks, not 54**.
The `COLLAPSED` and no-signal findings are unaffected. See "What we got wrong".
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
- **`n`** — the number of goal units the reward stream **never credits at any point** in
  the episode, read from `next.reward`.

`n > 0` in every episode of a task means the reward under-fires: the goal was reached, and
the reward never said so. That is a property of **every** episode of the task, not a
single trace, which makes it stronger evidence than one hdf5.

The distinction between "never credited" and "credited, then lost before the end" is the
whole report. A demo that satisfies a predicate and then breaks it again before
terminating ends below the goal without any reward defect. Conflating the two is the error
that produced the earlier 87-task claim.

## Method

1. `D = 1 / min|next.reward|` over all non-zero reward samples of the task. Every
   observed magnitude is then checked to be an integer multiple of `1/D`; if any is not,
   `D` is wrong and the task is not reported here.
2. `s0` — goal literals already true at reset, read from the BDDL `:init` block, capped
   per episode at `D` minus the units the trace credits.
3. Rollback repair. The demo collector rolls the simulator back and replays, and the
   reward function then emits a debit for credit it never issued in this trace. Any
   negative that would drive the running sum below `-s0/D` is zeroed — no real predicate
   flip can un-satisfy a unit that was never satisfied. A negative that stays at or above
   that floor is a genuine flip-back and is kept.
4. Track the satisfied count forward from reset: `satisfied_t = s0 + D * cumsum_t`.
   Then `n = D - max_t(satisfied_t)` — units never credited anywhere in the episode — and
   separately `D - satisfied_final` — units missing at the end, which includes anything
   earned and lost.
5. `m` from the BDDL goal literals.

Step 4 is the correction. The earlier draft measured only `k = D * sum(repaired rewards)`,
the end-of-episode total, and read `k < D` as under-firing. It is not: on 49 of the 100
tasks it is a demo that stopped short.

Two distinct defects fall out, and they need different fixes:

| Defect | Signature | Consequence |
|---|---|---|
| **DEAD_UNITS** | `D` is right, but `n > 0` in **every** episode — no demo in 200 ever shows the whole goal | reward omits credit it should have issued |
| **COLLAPSED** | `D < m` — the reward's own unit count is below the goal size | part of the goal is invisible to the reward at any credit level |

A third category is **not** a defect and is reported separately: `ENDS_SHORT`, where some
demos reach the whole goal and others stop a unit or two before terminating. The reward is
sound on those tasks; the demo corpus is uneven.

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

Raw event counts per episode: 1–4 positive rewards and 1–3 negative rewards. **Net credit
is 1 unit of 14 in 198 of 200 episodes**, and — the number that matters — **12.99 of 14
units are never credited at any point in the mean episode. 0 of 200 episodes reach the
whole goal.**

So: 8 plates are placed in a cabinet, the demo terminates successfully, and the reward
credits one unit. This case is unaffected by the correction below; it is under-firing on
either measurement, and it is the strongest case in the report.

## Case 2 — tasks that emit no reward at all

`rearranging_kitchen_furniture`: **200 of 200 episodes terminate, and `next.reward` is
identically zero in every frame of every episode.** There is no reward magnitude, so `D`
is not even measurable. Its BDDL goal is 3 `inside(..., cabinet.n.01_1)` literals plus one
already-true `not open(cabinet)`, so `m = 3` and `k = 0`.

## Case 3 — `make_microwave_popcorn` (collapsed, and it looks clean)

BDDL goal is two literals, **both false at init**: `real(cooked__popcorn.n.01_1)` and
`contains(popcorn__bag.n.01_1, cooked__popcorn.n.01_1)`. **`m = 2`.**

Measured across all 200 episodes: the only reward magnitude is `1.0`, so **`D = 1`**. Every
episode fires exactly one positive reward and terminates: 1 unit credited of `D = 1`,
nothing true at reset, 200 of 200 episodes reaching the whole of what the reward can
express, validity 1.000.

This task passes every internal consistency check that exists. It is nonetheless wrong:
the reward has one unit to spend on a two-predicate goal, so **half the goal is invisible
to the reward**, and no amount of episode data reveals it. This is the case that most
needs an organizer fix, and the one most likely to be missed — it was recommended to
entrants as a good task.

## What we got wrong, and what survives

An earlier draft of this report claimed **87 of 100 tasks** were defective, on the strength
of a bucket called `UNDER_FIRES` holding 54 of them. That bucket was measured as
`k = D * sum(repaired rewards) < D` — the end-of-episode credit total. It does not
distinguish a reward that never fired from a demo that stopped before finishing, and on 49
of those 54 tasks it was the second.

The tell was there and we missed it: on `chop_an_onion` all four units register in 200 of
200 episodes, and the episodes that end at 3 of 4 reach their third unit at t = 0.943 —
the moment the episode ends — while episodes that finish reach their fourth at the same
0.943. Same trajectory, cut at different depths. Grouping by how far each demo got makes
this immediate, and no amount of looking at end-of-episode totals does.

`wash_dog_toys` is the sharpest illustration, because it appeared in our worst-offenders
table at `phi0 = 0.333`. Its two uncredited units are exactly the two BDDL literals marked
true at initial state — `not covered(teddy_1, dust)` and `not covered(teddy_2, dirt)` —
which start satisfied and are never broken, so they correctly emit nothing. Reading the
`:init` block, which we had, would have removed it from the list.

**The 87 number would have been cheap for the organizers to refute and hard for us to
defend.** The corrected claim is smaller and holds.

What survives unchanged:

- **`COLLAPSED` (31 tasks).** Read from the BDDL goal size against `D`, never from `phi0`.
  Case 3 below is the archetype.
- **No reward signal (2 tasks).** `next.reward` identically zero.
- **`putting_dishes_away_after_cleaning`.** Under-fires on either measurement.
- **The sampling note.** A property of the data.

## Corpus-wide result

Measured over **all 20,000 episodes of all 100 tasks** (no sampling). Of 100 tasks:

| Verdict | Tasks |
|---|---|
| **CONSISTENT** — `D` equals the must-flip count *and* every valid demo reaches the whole goal | **14** |
| **ENDS_SHORT** — `D` right, instrumentation sound, some demos stop short of the goal | **49** |
| **DEAD_UNITS** — `D` right, but **no** demo in 200 ever shows the whole goal | **4** |
| **COLLAPSED** — `D` is below the number of BDDL predicates that must flip | **31** |
| **no reward signal at all** | **2** |

So **37 of 100 tasks** have a reward stream that cannot be reconciled with their own BDDL
goal — 4 under-firing, 31 collapsed, 2 silent — and **63 are safe to build a progress
label from**. `ENDS_SHORT` is a property of the demo corpus, not of the reward, and is not
part of the defect claim.

The 2 with no signal are `rearranging_kitchen_furniture` and `storing_food`: `next.reward`
is identically zero across 200 of 200 episodes, all of which terminate.

### The four under-firing tasks

Every one of these credits fewer than `D` units in **all 200** episodes; not one episode
in any of them reaches the whole goal.

| Task | `D` | must flip `m` | units never credited | episodes reaching the goal |
|---|---|---|---|---|
| `putting_dishes_away_after_cleaning` | 14 | 8 | **12.99** | 0 of 200 |
| `assembling_gift_baskets` | 16 | 16 | 12.13 | 0 of 200 |
| `sorting_vegetables` | 13 | 13 | 8.90 | 0 of 200 |
| `hiding_Easter_eggs` | 9 | 9 | 3.40 | 0 of 200 |

### Suspected, not claimed

Seven further tasks have sound instrumentation by the strict test — at least one demo does
reach the whole goal — but so few that the reward may still be under-firing intermittently.
We are not claiming these as defects; they are listed so the organizers can look:

| Task | `D` | units never credited (mean) | episodes reaching the goal |
|---|---|---|---|
| `tidying_bedroom` | 3 | 1.07 | 2 of 200 |
| `getting_organized_for_work` | 10 | 3.07 | 3 of 200 |
| `putting_up_Christmas_decorations_inside` | 9 | 1.96 | 6 of 200 |
| `canning_food` | 10 | 2.50 | 9 of 200 |
| `can_meat` | 9 | 1.96 | 12 of 200 |
| `thawing_frozen_food` | 7 | 1.84 | 18 of 200 |
| `cleaning_up_plates_and_food` | 7 | 1.12 | 19 of 200 |

### Collapsed denominators (these look clean on every internal check)

Diagnosed from `D` against the BDDL goal alone. No per-episode statistic can see this, so
none is quoted: these tasks look clean on every internal check.

| Task | `D` | must flip `m` | goal literals | goal invisible to the reward |
|---|---|---|---|---|
| `cook_brussels_sprouts` | 3 | 24 | 26 | 21 of 24 |
| `setup_a_bar_for_a_cocktail_party` | 9 | 20 | 20 | 11 of 20 |
| `sorting_bottles_cans_and_paper` | 3 | 16 | 16 | 13 of 16 |
| `cook_broccolini` | 3 | 11 | 11 | 8 of 11 |
| `setting_the_table` | 4 | 8 | 8 | 4 of 8 |
| `laying_tile_floors` | 2 | 8 | 8 | 6 of 8 |
| `putting_away_toys` | 1 | 8 | 8 | 7 of 8 |
| `chopping_wood` | 4 | 8 | 8 | 4 of 8 |
| `collecting_childrens_toys` | 6 | 7 | 7 | 1 of 7 |
| `sorting_books_on_shelf` | 6 | 7 | 11 | 1 of 7 |

### A sampling note the organizers may also want

`D = 1/min|reward|`, so it is set by the single smallest reward magnitude anywhere in the
task. On `wash_dog_toys` the magnitude `1/6` occurs **exactly once in 200 episodes**. Any
analysis that reads a subset of episodes will measure `D = 3` and be wrong by a factor of
two. This is a property of the data, not of our tooling, and it means per-task reward
statistics computed on anything less than the full episode set cannot be trusted.


## What would fix it

For the four under-firing tasks the reward function is not crediting every satisfied
predicate — most visibly where a quantified goal (`forall` over 8 plates) collapses to a
single credit. For `COLLAPSED` tasks the denominator itself is wrong: `D` should equal the
number of goal literals that must flip, and where it does not, the reward cannot represent
the goal it is scoring.

Two checks the organizers could run in CI, needing no simulator, on numbers already in the
released parquet and the BDDL:

1. **`D == m`** for every task. Catches all 31 `COLLAPSED` tasks, and is the check that
   cannot be replaced by anything measured from the reward stream.
2. **At least one demo per task reaches `D` credited units**, tracking the satisfied count
   forward from the BDDL initial state. Catches the 4 under-firing tasks.

**Do not use "every successful demo credits `k == D`" as the check.** That is what we ran,
and it fails on 49 tasks whose rewards are correct, because a demo that satisfies a
predicate and then breaks it again — or simply stops early — ends below `D` with nothing
wrong. It also fails on any task with goal literals true at reset, which emit no delta and
so never appear in `k` at all.
