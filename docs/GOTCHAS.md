# Gotchas that will cost you a day each

Split out of the README.

---


- **The pinned tag moves.** It was `v3.9.1`, now `v3.9.2`. Check the
  [evaluation page](https://behavior.stanford.edu/challenge/evaluation.html) before every
  fresh clone and again before submitting.
- **Baseline checkpoints exist for one task only** (`turning_on_radio`). There is no
  pretrained 100-task baseline to run — a real baseline number means training one.
- **First OmniGibson import takes ~5 minutes.** One-time shader compile, not a hang.
- **Hang at `HydraEngine rtx failed creating scene renderer`** → `export OMNIGIBSON_GPU_ID=0`.
- **CuRobo often fails to build.** Install without `--primitives` first.
- **PyPI packages and Docker install are unavailable** during their monorepo migration.
- **Each parallel eval worker needs its own policy server port.** Worker `i` uses
  `base_port + i`. The IP-submission mode requires ≥50 ports for exactly this reason.
