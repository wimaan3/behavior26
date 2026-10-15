# Radio submission run: public test instances 301–320

**Arm B (progress head), `turning_on_radio`, one attempt per public test instance:
9 successes out of 20 (Q = 0.45).** Successful instances: 304, 306, 307, 308, 309, 310, 316,
317, 320.

## How it was run

- Run on **13 Oct 2026, 05:42–08:01 UTC**, from `docs/HANDOFF-2026-10-11.md` §5:
  `SUBMISSION=1 ARMS=B EXPERIMENT=configs/experiments/003-radio-submission.yaml`, checkpoint
  `/workspace/shot2/checkpoints/pi05_b1k_frozen_vlm_progress/armB/5999`, RTX PRO 4500.
  An earlier attempt the same morning crashed in the simulator driver
  (`eval.crash-driver595.log`) and was relaunched; the relaunch completed all 20.
- This session did not launch that run; it found the finished results on the volume on
  14 Oct while starting the same run, and did **not** run the test set a second time.
  One run, one result.
- Arm B was chosen before the run because it was nominally ahead on the development
  instances (40/108 against 38/108); the A/B difference is not significant, so either arm is
  an equally fair entry.

## Files

- `results/`: the 20 per-attempt JSON files.
- `eval.log`, `evaluator_armB_turning_on_radio.log`, `serve_armB_turning_on_radio.log`: the run.
- `behavior26_radio_public_package.zip`: the package from `python -m submission.build`
  (team `behavior26`, affiliation `3MagicLabs`, standard track, public test set, intended task
  `turning_on_radio`, wrapper `omnigibson.eval.wrappers.RGBDFullResWrapper`, BEHAVIOR-1K v3.9.2).
- Videos (20) are on the owner's laptop in `~/behavior26-results/submission2/submission2/armB/videos`,
  not in this repository. The portal needs a shareable link to them.

## What the score means

The leaderboard averages Q over all 100 tasks; the 99 not submitted count as zero, so this
entry's overall score is about **0.0045**. On its one task it succeeds on 45% of the public
test instances, against 37% on our development instances and the released checkpoint's 11%
(different instances, prompt and hardware, so only indicative).

## Still to do (owner)

1. Confirm the challenge rules allow an entry fine-tuned from the organisers' released checkpoint.
2. Upload the zip through the challenge portal with a shareable link to the 20 videos.
