# Testing the whole pipeline without a GPU

Split out of the README. This is load-bearing: GPU time is rented and
billed, so anything catchable here must be caught here.

---


GPU time is the scarcest resource here, and a bug found on a rented card is a bug paid
for twice. Two pieces make the entire chain runnable on a laptop in seconds:

| Piece | What it is |
|---|---|
| `policy/null_server.py` | Websocket server returning all-zero actions. No model, no GPU. Also a real baseline — a null policy scores ~0.093, because Q is scored on the final state and many tasks start with predicates already satisfied. |
| `tests/mock_evaluator.py` | Speaks the real evaluator's CLI and websocket protocol and writes rollout JSONs in the real schema. No simulator. |

`harness/launch.py --eval-module` swaps the mock in for the real evaluator, so the whole
chain runs unchanged:

```bash
# terminal 1 — one null server per eval worker
python -m policy.null_server --base-port 8000 --num-servers 3

# terminal 2 — the full chain, ~2 seconds
python -m harness.launch --config configs/experiments/900-mock-smoke.yaml \
    --eval-module tests.mock_evaluator --workers 3 --base-port 8000 \
    --extra-eval-arg=--fast --output-dir rollouts/mockA
python -m analysis.parse    rollouts/mockA --universe 100 --per-task
python -m analysis.failures rollouts/mockA --sample 2
python -m submission.build  --rollouts rollouts/mockA \
    --wrapper tests/fixtures/mock_wrapper.py \
    --robot-config tests/fixtures/mock_robot.yaml

pytest tests/test_pipeline.py -q
```

**This proves plumbing and schema, never behaviour.** The Q numbers it produces are
synthetic. A green run says the pipeline is wired correctly, not that the policy is good.

The protocol is implemented against the published docs and is **not yet verified against
a BEHAVIOR-1K checkout** — see the assumption list at the top of `policy/wire.py` and
confirm all of it on the first real run.
