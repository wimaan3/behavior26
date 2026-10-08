"""scripts/shot_two_chain.sh -- training -> evaluation -> comparison on one pod, unattended.

Executed against a stub evaluator: nobody is connected when this runs, so each branch of the
plan (docs/sessionE-2026-10-08-shot2/PLAN.md) is driven for real here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "shot_two_chain.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# A stand-in for eval_arms.sh: records how it was called, writes Q_A / Q_B for every instance
# it was asked for (4 attempts each), and ends with the status it is told to.
STUB = r'''#!/usr/bin/env bash
echo "CALL override=[$INSTANCES_OVERRIDE] hours=$MAX_HOURS ckA=$CKPT_A ckB=$CKPT_B exp=$EXPERIMENT asset=$ASSET_ID" >> "$OUT/calls.txt"
ids="${INSTANCES_OVERRIDE:-10 11 12 13 14 15 16 17 18 19 20}"
for arm in A B; do
  mkdir -p "$OUT/arm$arm/json"; q=$( [ $arm = A ] && echo "$STUB_QA" || echo "$STUB_QB" )
  for i in $ids; do for r in 0 1 2 3; do
    echo "{\"q_score\": {\"final\": $q}}" > "$OUT/arm$arm/json/turning_on_radio_${i}_$r.json"
  done; done
done
echo "paired comparison" > "$OUT/compare.txt"
printf '%s\nTERMINAL\n' "${STUB_STATUS:-DONE}" > "$OUT/STATUS"
'''


def _chain(tmp_path, train_status="DONE 2026", qa="0.0", qb="0.0", stub_status="DONE", ckpts=True):
    b26, train, ev, run = (tmp_path / n for n in ("b26", "shot2", "eval2", "chain2"))
    (b26 / "scripts").mkdir(parents=True); ev.mkdir()
    (b26 / "scripts" / "eval_arms.sh").write_text(STUB)
    train.mkdir()
    (train / "STATUS").write_text(f"{train_status}\nTERMINAL 2026\n")
    if ckpts:
        for cfg, arm in (("pi05_b1k_frozen_vlm", "armA"), ("pi05_b1k_frozen_vlm_progress", "armB")):
            for step in ("999", "5999"):
                (train / "checkpoints" / cfg / arm / step / "params").mkdir(parents=True)
    env = dict(os.environ, RUN=str(run), TRAIN=str(train), EVAL=str(ev), B26=str(b26), POLL="1",
               STUB_QA=qa, STUB_QB=qb, STUB_STATUS=stub_status)
    p = subprocess.run(["bash", str(CHAIN)], env=env, capture_output=True, text=True, timeout=60)
    calls = (ev / "calls.txt").read_text().splitlines() if (ev / "calls.txt").exists() else []
    return (run / "STATUS").read_text().splitlines(), calls, run


def test_failed_training_is_never_evaluated(tmp_path):
    status, calls, _ = _chain(tmp_path, train_status="HEALTH_FAIL warm start: action_loss opens at 0.9")
    assert status[0].startswith("TRAIN_FAILED: HEALTH_FAIL") and status[-1].startswith("TERMINAL")
    assert calls == []


def test_missing_checkpoints_stop_before_the_evaluator(tmp_path):
    status, calls, _ = _chain(tmp_path, ckpts=False)
    assert status[0].startswith("NO_CHECKPOINT") and calls == []


def test_the_first_block_runs_on_nine_instances_with_the_latest_checkpoints(tmp_path):
    status, calls, _ = _chain(tmp_path, qa="1.0")
    assert "override=[10 11 12 13 14 15 16 17 18]" in calls[0]
    assert "armA/5999 " in calls[0] and "armB/5999 " in calls[0]         # 5999, not 999
    assert "exp=configs/experiments/002-radio-ab.yaml" in calls[0] and "asset=turning_on_radio" in calls[0]


def test_both_arms_at_zero_stops_after_the_first_block(tmp_path):
    status, calls, _ = _chain(tmp_path, qa="0.0", qb="0.0")
    assert status[0].startswith("STOPPED_NO_SUCCESS: arm A 0/36, arm B 0/36")
    assert len(calls) == 1 and status[-1].startswith("TERMINAL")


def test_one_arm_succeeding_is_enough_to_continue_to_the_full_run(tmp_path):
    """The stopping rule is 'the policy is broken', never 'the difference looks bad'."""
    status, calls, run = _chain(tmp_path, qa="0.0", qb="1.0")
    assert len(calls) == 2 and "override=[]" in calls[1]
    assert status[0].startswith("DONE: arm A 0/44, arm B 44/44")
    assert (run / "compare.txt").read_text().strip() == "paired comparison"


def test_an_incomplete_evaluation_is_reported_not_called_done(tmp_path):
    status, calls, _ = _chain(tmp_path, qa="1.0", stub_status="CAP_HIT after 7 h")
    assert status[0].startswith("EVAL_BLOCK_FAILED: CAP_HIT")


def test_the_chain_carries_no_credential_and_cannot_claim_to_stop_the_pod():
    t = CHAIN.read_text()
    code = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("#"))
    for word in ("RUNPOD_API_KEY", "runpodctl", "git push", "token"):
        assert word not in code, word
