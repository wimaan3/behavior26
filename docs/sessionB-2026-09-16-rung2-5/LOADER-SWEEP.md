# Loader sweep, 2026-09-16: what we got and what we lost

**Status: not answered. The sweep result was lost with the pod.**

## What happened

| UTC | Event |
|---|---|
| 06:48 | Backend comparison started (`loader_bench.py`, 24 workers, 70 batches) |
| ~07:03 | pyav row read: "49.0 items/s, KEEPS UP". This is **not trustworthy**, see below. Killed before torchcodec finished |
| 07:05 | 90-step training probe at 24 workers launched. Killed at ~step 12: 90 steps is still inside a 72-batch prefetch fill, so it could not answer the question |
| 07:15 | Corrected sweep started: `loader_bench.py --batches 200 --workers 8 16 24`, pyav then torchcodec. Expected about 1.7 h |
| 07:38 | Last check: still on the first config (8 workers). No rows printed yet |
| ~13:45 | Pod `5cm27uy401j5j3` terminated (last billing hour: 13:00–14:00, partial) |

The sweep wrote to `/opt/sessionb2/loader/sweep.log` on **container disk**. Nobody copied it
off before the pod went away, so the result is gone. The pod also billed about 6 h after my
last check, well past the sweep's expected end (~09:00).

**The process failure:** I left an expensive pod running unattended with its only output on
disk that would be wiped. The session's background watchers stopped when the conversation
ended, so nothing salvaged the log or stopped the pod. Rule going forward: any pod job that
outlives the conversation must (a) write results somewhere that survives the pod, and
(b) stop the pod itself when it finishes (`runpodctl stop pod $RUNPOD_POD_ID` as the last line).

## Cost

- Pod `5cm27uy401j5j3` total: **$11.47** (session B rungs 2–5 plus loader work)
- Of that, ~$4.30 is the six hours after my last check that produced nothing we kept
- **Project total: $23.61 of $200** (from the RunPod billing API, pods only; the network
  volume `96mu3d0s32` bills separately)

## What we still know (from the committed rung-4 log, not the lost sweep)

The rung-4 finding is unaffected. It comes from `logs/train_rung4_armA.log`, which is in git:

- 39 stalls, **exactly every 24 steps**, ~103 s each, **52.8% of wall clock**
- sustained loader rate **4.0 samples/s** at 8 workers vs **8.2 needed**
- arithmetic says ~2.1× the workers (≈17) removes the stall. That is a **prediction, not yet measured**

## The one number from today to ignore

The 49 items/s pyav row at 24 workers came from the *old* bench: depth assumed workers×2,
and 70 batches is only 21 past the buffer, far too few. Commit `eebe995` fixed both.
The fixed bench now refuses to report a rate from a run that short.

## To re-run (≈ $0.60–1.25)

On a training pod with the merged coffee root, container disk only:

```bash
OUT=/workspace/loader_sweep   # the VOLUME, so it survives the pod
mkdir -p $OUT
/opt/openpi/.venv/bin/python -u scripts/loader_bench.py \
  --dataset-root /opt/merged/set_up_a_coffee_station_in_your_kitchen \
  --repo-id set_up_a_coffee_station_in_your_kitchen \
  --batch-size 32 --batches 200 --workers 8 16 24 > $OUT/sweep.log 2>&1
runpodctl stop pod $RUNPOD_POD_ID
```

A cheaper option: skip the microbenchmark and run shot one's first ~500 steps at
`WORKERS=24`. Then run `python -m analysis.rung_report` on that log. `stalls()` gives the
answer directly. If `period` is None and clock share is ~0, the sawtooth is gone.
