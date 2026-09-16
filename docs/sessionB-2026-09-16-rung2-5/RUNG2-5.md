# Session B rungs 2-5 -- results

Rules fixed in AB_PROTOCOL revision 2026-09-15 before the run (window 100, steady state from step 50).

## Rung 2 -- Branch 0 manipulation check

**LEARNING** -- decreasing by >2 SE
progress_loss first 100: 0.6971, last 100: 0.6559, drop +0.0412, z = 18.7

## Rung 2 -- action loss with the head present (paired, same batches)

**NOT_DEGRADED** -- mean B - A over last 100 steps +0.0007, 95% CI [-0.0002, +0.0016]

## Rung 3 -- lambda calibration (gradient share)

- steps >=    0 (n=40): median |g_a|/|g_p| 0.603; share at lambda=0.1 14.2%; lambda for 10/20/30% = 0.067 / 0.151 / 0.258; median cos +0.011
- steps >=   50 (n=38): median |g_a|/|g_p| 0.564; share at lambda=0.1 15.1%; lambda for 10/20/30% = 0.063 / 0.141 / 0.242; median cos +0.007
- steps >=  940 (n=2): median |g_a|/|g_p| 0.685; share at lambda=0.1 12.7%; lambda for 10/20/30% = 0.076 / 0.171 / 0.293; median cos +0.018

If these three windows disagree sharply, lambda is still drifting -- report that rather than pin lambda to an early transient (AB_PROTOCOL session-B note).

## Rung 4 -- throughput (arm A)

median 3.90 s/step, **mean 8.11 s/step** (p90 3.98) over 949 intervals; median GPU utilisation 0.0%

Step time is bimodal: a fast step plus a periodic loader stall. Cost follows the MEAN.
- 10,000 steps ~ 22.5 h ~ $16.23 per arm at $0.72/h
- 30,000 steps ~ 67.6 h ~ $48.68 per arm at $0.72/h

### Where the time goes

39 intervals over 20 s (median 104 s) account for **52.8% of the wall clock**.

They land **exactly every 24 steps** -- a prefetch sawtooth, not slow data. 24 is the pipeline depth (workers x prefetch factor): the trainer drains a full queue at the GPU-bound rate, then blocks while the workers refill it.

Sustained loader rate **4.0 samples/s** against the **8.2 samples/s** a 3.90 s step consumes -- so roughly **2.1x** more workers removes the stall and floors step time at 3.90 s.
  - if fixed: 10,000 steps ~ 10.8 h ~ $7.80 per arm
  - if fixed: 30,000 steps ~ 32.5 h ~ $23.40 per arm

## Rung 5 -- does steps/s depend on task count?

one task 0.256 steps/s, two tasks 0.256 steps/s, ratio 1.00. Within ~10% means batch-bound as expected, so k is a CONVERGENCE question that throughput cannot settle.

