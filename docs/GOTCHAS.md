# Gotchas that will cost you a day each

Split out of the README.

---


- **The pinned tag moves.** It was `v3.9.1`, now `v3.9.2`. Check the
  [evaluation page](https://behavior.stanford.edu/challenge/evaluation.html) before every
  fresh clone and again before submitting.
- **Baseline checkpoints exist for one task only** (`turning_on_radio`). There is no
  pretrained 100-task baseline to run — a real baseline number means training one.
- ~~**First OmniGibson import takes ~5 minutes.**~~ **Not reproduced (2026-09-13).**
  Measured on the session-A pod: 45.0 / 44.5 / 44.1 s across three runs, GPU at
  0%, and `OMNIGIBSON_APPDATA_PATH` still empty afterwards — so no shader compile
  happened at import time on v3.9.2. If one occurs it is on `og.Environment(...)`
  creation. Budget ~45 s for an import; treat 5 minutes as a symptom, not normal.
- **Hang at `HydraEngine rtx failed creating scene renderer`** → `export OMNIGIBSON_GPU_ID=0`.
- **CuRobo often fails to build.** Install without `--primitives` first.
- **PyPI packages and Docker install are unavailable** during their monorepo migration.
- **Each parallel eval worker needs its own policy server port.** Worker `i` uses
  `base_port + i`. The IP-submission mode requires ≥50 ports for exactly this reason.
