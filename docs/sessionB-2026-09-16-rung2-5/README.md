# Session B, rungs 2–5 — what we learned

Pod: RTX PRO 4500 (Blackwell), 27 vCPU, 251 GB RAM, $0.72/h, EU-RO-1.
Rules were fixed in `AB_PROTOCOL.md` revision 2026-09-15 **before** the run.
Machine-generated results: [`RUNG2-5.md`](RUNG2-5.md). Raw logs: [`logs/`](logs/).

## The four questions, answered

| Rung | Question | Answer |
|---|---|---|
| 2 | Does the progress head learn anything at all? (Branch 0) | **LEARNING.** progress_loss 0.6971 → 0.6559 over 1000 steps, z = 18.7 |
| 2 | Does adding the head damage the action loss? | **NOT_DEGRADED.** paired B − A = +0.0007, 95% CI [−0.0002, +0.0016] |
| 3 | What λ puts the progress term at a chosen share of the gradient? | share ≈ **14–15 %** at λ = 0.1; **λ ≈ 0.14–0.15** for 20 %; median cos(g_a, g_p) ≈ **+0.01** |
| 5 | Does steps/s depend on how many tasks k we train on? | **No** — ratio 1.00 one-task vs two-task. k is a *convergence* question, not a throughput one |

Branch 0 matters because a flat or NaN progress loss would mean every downstream
A/B number is measuring nothing. It is not flat.

The λ answer is a *gradient-share* answer, not a performance answer: at λ = 0.1 the
progress term contributes about a seventh of the update. The near-zero cosine says the
two terms are close to orthogonal — the head is not fighting the action objective,
which is consistent with the action loss being undamaged.

## Rung 4 — the expensive finding

Arm A, 1000 steps, batch 32, 8 dataloader workers:

- median **3.90 s/step**, p90 3.98 s/step
- mean **8.11 s/step**
- median GPU utilisation **0 %**

Quoting the median would have underpriced shot one by about 40 %. The mean is what
the clock charges, so the report quotes mean, median and p90.

### The stall is not random — it is a prefetch sawtooth

Step intervals past step 50 (n = 950):

```
min 3.80   p50 3.90   p90 3.98   p99 108.9   max 113.1
```

39 intervals exceed 20 s. Their step numbers are 72, 96, 120, 144, … — **exactly every
24 steps**, each costing ~103 s. Those 39 stalls are **52.8 % of the entire wall clock**.

That period is the pipeline depth, not the data: 8 workers × prefetch depth 3 = 24
batches. The trainer drains a full queue at the GPU-bound rate (3.9 s/step for 24
steps = 94 s), then blocks ~103 s while the workers refill it.

So the sustained loader rate is 24 batches × 32 samples / 196.6 s = **3.9 samples/s**,
against the **8.2 samples/s** a 3.9 s step consumes. The loader is delivering under half
of what training needs, and the GPU is idle for half the run. Per worker that is
0.49 samples/s, i.e. ~2 s to assemble one sample — random access across six mp4 files.

### Why this was nearly missed twice

1. `analysis/rung_report.py` first reported the **median**, which is blind to a stall
   that lands on 4 % of the steps but eats half the clock. Fixed in `d07f58d`.
2. `scripts/loader_bench.py` first reported **777 items/s "KEEPS UP"** at 24 workers —
   it was timing the prefetch queue draining, not the loader producing. Fixed in
   `89e0df7` by discarding `workers × 2` batches before measuring. Even after that fix
   the microbenchmark still over-reads (49 items/s, n = 21) because 70 batches is not
   enough to outrun the buffer. **The training log is the trustworthy instrument** —
   a 1000-step run contains 39 independent refill cycles.

### What it costs, and the lever

At 8 workers, mean 8.11 s/step:

| steps | hours | $/arm at $0.72/h |
|---|---|---|
| 10,000 | 22.5 | $16.23 |
| 30,000 | 67.6 | $48.68 |

If raising the worker count removes the sawtooth, the step time floors at the
GPU-bound 3.90 s and the same runs cost:

| steps | hours | $/arm |
|---|---|---|
| 10,000 | 10.8 | $7.80 |
| 30,000 | 32.5 | $23.40 |

The arithmetic says the break-even is ~17 workers (8.2 / 0.49); the box has 27 vCPU and
218 GB free, so 24 workers fits. That prediction is being tested directly — see
[`W24.md`](W24.md).
