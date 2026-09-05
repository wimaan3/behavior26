# Evaluation: instance ids, scoring, and comparing runs

Everything about how the benchmark scores us, and how we measure a change
against it. Split out of the README, which now links here.

See also [AB_PROTOCOL.md](AB_PROTOCOL.md) for the frozen A/B design.

---

## Why the harness matters more than it looks

Verified against BEHAVIOR-1K v3.9.2,
`omnigibson/eval/utils/eval_utils.py`:

```python
NUM_TEST_INSTANCES        = 40
NUM_PUBLIC_TEST_INSTANCES = 20
NUM_HIDDEN_TEST_INSTANCES = 20
TEST_INSTANCE_IDS = list(range(301, 341))
```

**Instance ids are 301–340.** Public test is 301–320, hidden is 321–340, and
everything below 301 is a *training* instance. An earlier version of this
section said "report on 0–9, dev on 10–19" — those are training instance ids,
and the convention was wrong repo-wide.

| Set | Ids | Use |
|---|---|---|
| Training | < 301 | The only place to iterate. Dev loop lives here. |
| Public test | 301–320 | Reported. All 20 are scored. Never tune on these. |
| Hidden test | 321–340 | Never touch. |

A full public submission is **100 tasks × 20 instances × 1 rollout = 2,000
rollouts**. At the organizers' published throughput (~13.5 FPS full-res RGB+D)
plus 150–300s scene load per trial, one rollout is roughly **20–25 minutes**:

**2,000 rollouts ≈ 700–840 GPU-hours ≈ ~7–9 days on 4 cards, ~3.5–4.5 days on 8.**

All of it rented, so it is billed.

Two consequences that are easy to get wrong:

1. **There is no self-test split inside the public set.** `compute_final_q_score`
   averages over all 20 public instances per task, so holding back 10 of them as
   a "dev set" does not protect anything — you would be tuning on half the
   leaderboard and reporting the other half. The dev loop must use training
   instances. `harness/launch.py` refuses a `mode: train` config that names a
   test instance — **our rule, not the evaluator's**: upstream's
   `resolve_instance_ids` passes train-mode ids through unvalidated. We are
   deliberately stricter.
2. **Missing rollouts are zeros, not omissions.**
   `q_score_avg[task] = sum(...) / n_instances_per_task` with
   `n_instances_per_task = 20`. Submitting 10 instances per task does not score
   those 10 — it **halves** the reported score.

Experiment configs declare real instance **ids**; the harness converts them to
the `--instance-indices` the evaluator wants (which are indices into a split —
index 0 of `public_test` is instance 301). That indirection is exactly what hid
the old mistake, so the config layer no longer exposes it.

| Config | Scope | Cost | When |
|---|---|---|---|
| **Dev loop** | ~12 tasks × 3 training instances = 36 rollouts | ~13 GPU-hr, overnight | Every iteration |
| **Full eval** | 100 × 20 = 2,000 rollouts, ids 301–320 | 700–840 GPU-hr | Only when submitting |

---

## Scoring, and what it implies

Q = fraction of BDDL goal predicates satisfied **at episode end**, averaged over all 100
tasks. Two consequences worth internalising:

1. **"At episode end" is literal.** Complete the task, then knock the object back out
   while closing a door, and you score zero. Stability matters as much as achievement —
   which is why `early_stop_on_goal` is in the experiment config.
2. **Broad shallow competence beats narrow depth.** 20 tasks at Q=0.5 with 80 zeros gives
   mean 0.10. Every task at Q=0.15 gives 0.15. Reliably completing the *first* predicate
   of all 100 tasks outscores perfectly solving fifteen.

---

## Comparing two runs: always pair

The evaluator is nondeterministic and we run one rollout per instance. Comparing two
*independent* 36-rollout sweeps needs roughly **ΔQ > 0.115** to clear the noise — wider
than the gap between 1st and 5th place in 2025. An unpaired A/B at dev-loop size cannot
resolve the differences we care about.

Run both arms on the **same instances** and difference per instance instead. The
instance-to-instance variance — most of the total, because some instances are simply
harder — cancels, and the detectable difference drops to about **0.033**. Same GPU spend,
~3.5x the resolution.

```bash
python -m analysis.compare rollouts/baseline rollouts/candidate --per-instance
```

It reports mean ΔQ, standard error, a 95% CI on the paired difference, and what the same
rollouts would have resolved unpaired. If the two runs do not cover identical
`(task, instance_id, rollout_id)` keys it warns loudly and refuses to report — a broken
pairing throws away the entire advantage. This is why the dev subset stays frozen.
