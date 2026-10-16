# BEHAVIOR 2026 Challenge Submission

Package: `standard.public.behavior26.Elmdin.20261016`
Track: **standard** · Test set: **public** · Team: **behavior26** (Elmdin)

## What this is

A π₀.₅-based policy served over the websocket protocol the BEHAVIOR-1K evaluator
speaks. `json/` holds 20 unmodified per-rollout metrics files.

## How to evaluate this policy

### 1. Start the policy server

The Docker image serves the policy over the websocket protocol described below, within a single 24GB VRAM GPU.

- Image / endpoint: wimaan3/behavior26-policy:armB-5999@sha256:84f7bd76d2edc28eeb88c4a16dd119d7c6fd3d5de55d77b85a46bd3ba904a73e

The server must implement the protocol in
`omnigibson/eval/utils/network_utils.py`:

- answer `GET /healthz` with 2xx on the same host:port;
- on websocket connect, send **one msgpack metadata frame first** — the client
  blocks on `unpackb(conn.recv())` before sending anything;
- for each observation frame, reply `{"action": <ndarray>}`. The value must
  decode to a real `numpy.ndarray` (the client calls `th.from_numpy` on it), and
  its width must equal `robot.action_dim` (23 for R1Pro);
- ndarrays use the evaluator's own msgpack extension
  (`{b"__ndarray__": True, b"data", b"dtype", b"shape"}`), **not** msgpack-numpy;
- a `{"reset": True}` frame expects **no reply**.

### 2. Run the evaluator

```
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 \
  --port 8000 \
  --mode public_test \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --num-rollouts 1 \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --robot-config r1pro.yaml \
  --output-dir outputs/eval \
  --write-video
```

`--instance-indices` are **indices into the split**, not instance ids. For
`--mode public_test` they index `TEST_INSTANCE_IDS[:20]`,
i.e. instances **301–320**. Result filenames carry the resolved id.

### 3. Score

```
python -m omnigibson.eval.utils.score_utils --input-dir standard.public.behavior26.Elmdin.20261016 --output-dir scores/
```

The directory name is parsed as `<track>.<testset>.<team>.<affiliation>.<date>`,
which is why the package is laid out that way.

## Contents

| File | Purpose |
|---|---|
| `json/` | 20 rollout metrics files, unmodified |
| `rgbd_full_res_wrapper.py` | Evaluation wrapper. Exposes RGB, depth and proprioception only. |
| `r1pro.yaml` | Exact robot config used for evaluation |


## Rollout videos

https://drive.google.com/drive/folders/1HBUMYrx29VgiCbJynF9re3L8EhgbWvWz

MP4s are not zipped; they are submitted as a link through the portal.

## Observation compliance

The wrapper exposes only RGB, depth and proprioception. No ground-truth
segmentation, object state, target object pose, full-scene point cloud, or robot
global pose is read from the simulator at evaluation time. The environment is not
manipulated directly — no teleporting the robot and no setting object states.

## Notes and disclosures

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
docker run --gpus all -p 8000:8000 wimaan3/behavior26-policy:armB-5999@sha256:84f7bd76d2edc28eeb88c4a16dd119d7c6fd3d5de55d77b85a46bd3ba904a73e
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
