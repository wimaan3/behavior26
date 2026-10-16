**Scope.** This is a partial submission: one task, `turning_on_radio`. The policy was trained
only for this task; the other 99 tasks are not submitted.

**Model and provenance.** π₀.₅ (openpi, `wensi-ai/openpi` at commit
`0cc8e355f7bac0976db1cc3139b1ff0379feea60`, plus our patches in the repository below). We
**fine-tuned from the organisers' released `turning_on_radio` checkpoint (`pi05_b1k`)**: 6,000
steps on the `turning_on_radio` demonstrations of the 2026 challenge dataset, batch 32, with the
vision-language backbone frozen and an auxiliary progress-prediction head (config
`pi05_b1k_frozen_vlm_progress`, loss weight 0.15). Norm statistics are the released checkpoint's
own. Prompt: the task name, `turning_on_radio`. Code, configs and training logs:
https://github.com/wimaan3/behavior26

**Evaluation run.** BEHAVIOR-1K v3.9.2, the stock evaluator (`omnigibson.eval.eval`) and the
stock `omnigibson.eval.wrappers.RGBDFullResWrapper`; simulator settings and evaluation scripts
unmodified. One rollout per public test instance (301–320), on one NVIDIA RTX PRO 4500 (32 GB),
13 Oct 2026.

**Rerun after an infrastructure crash.** The first launch of this evaluation (13 Oct, 05:18 UTC)
landed on a cloud host with NVIDIA driver 595.91, where the simulator process crashed before
finishing any rollout; its logs are kept (`eval.crash-driver595.log`,
`evaluator_armB_turning_on_radio.crash-driver595.log` in the repository). It was relaunched
with the same policy and settings on a host with driver 580.159 and completed all 20 instances
(05:42–08:01 UTC); those 20 results are the ones submitted.

**Other runs on the public test set, for transparency.** After this entry's run, our control
policy (the same fine-tune without the progress head) was evaluated once on the same 20
instances for an internal comparison (also 9/20). It is not submitted and was not used to
choose this entry, which was fixed before either run.

**Serving.** The policy server is openpi's `scripts/b1k/serve_b1k.py`, the same server that
produced these results. It listens on port 8000, answers `GET /healthz`, sends the metadata
frame first, and returns 23-dimensional actions (R1Pro). Docker image:

```
docker run --gpus all -p 8000:8000 IMAGE_REFERENCE
```

then point the evaluator at `--host <machine> --port 8000`. The server takes several minutes
to load before `/healthz` answers. One container serves one rollout at a time.

What has been checked on the image itself: on a CPU-only machine the container starts, restores
the checkpoint and norm statistics, and answers `/healthz` with 200. A full rollout from the
image on a GPU has not been run; the model, server code and openpi commit inside it are the
ones that produced the results above. Please contact us if anything fails to start.

**Hardware.** One NVIDIA GPU; the model uses about 6 GB of GPU memory (24 GB is ample). It
computes in **bfloat16**, so it needs a GPU with bfloat16 support (Ampere or newer, e.g. RTX
3090 / A5000). It has not been tested on a Titan RTX.
