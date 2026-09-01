# HANDOFF — GPU-free test harness

Branch `local/test-harness`, 6 commits, **not pushed**.

## Done

| Task | Result |
|---|---|
| 1. Null policy server | `policy/null_server.py` + `policy/wire.py`. Zero actions, healthz, standalone. |
| 2. Mock evaluator | `tests/mock_evaluator.py`. Real CLI, real protocol, real output schema, no simulator. |
| 3. Paired evaluation | `analysis/compare.py`. Mean ΔQ, SE, 95% CI, coverage guard. |
| 4. LICENSE | MIT, 2026. **Confirm the copyright holder name.** |
| 5. End-to-end proof | Full chain runs in ~2s. 4 bugs found and fixed. 16 tests passing. |

## Verified working (no GPU)

`null_server → mock_evaluator → harness/launch.py → parse → failures → compare → build`

```bash
python -m policy.null_server --base-port 8000 --num-servers 3 &
python -m harness.launch --config configs/experiments/900-mock-smoke.yaml \
    --eval-module tests.mock_evaluator --workers 3 --base-port 8000 \
    --extra-eval-arg=--fast --output-dir rollouts/mockA
pytest tests/test_pipeline.py -q
```

Also verified: `--resume`, action_dim mismatch rejection, dead-server handling,
chunked-action servers, coverage-mismatch refusal.

## Bugs fixed (all in pre-existing code)

1. `harness/launch.py` `already_done()` — substring filename matching reported
   never-run instances as complete. **Silent zero-scoring on a resumed full sweep.**
2. `analysis/failures.py` — `CRASHED` unreachable (parse defaulted missing q to 0.0).
3. `analysis/failures.py` — `TIMEOUT` unreachable (`timeout_ratio` never referenced).
4. `analysis/failures.py` — `x or 0.0` does not default NaN (NaN is truthy).

All four funnelled bad rollouts into `ACTIVE_NO_PROGRESS`.

## Pending / next steps

- **Verify the protocol against a real BEHAVIOR-1K checkout.** See the assumption list
  at the top of `policy/wire.py`: msgpack-numpy, the metadata frame, the `action` vs
  `actions` key, and the rollout filename convention. Each is a flag, not a hardcode.
- **`policy/wrapper.py` does not exist** — required submission artifact.
- **`configs/robot/r1pro.yaml` does not exist** — required submission artifact.
  `tests/fixtures/` holds test stand-ins for both. Do not submit them.
- `configs/experiments/001-dev-loop.yaml` still has `tasks: []`.
- `pip install msgpack-numpy` before the first GPU run.
- `analysis/census/*` has uncommitted CRLF-only churn, untouched by this work.
  Consider a `.gitattributes` with `* text=auto` to stop it recurring.
