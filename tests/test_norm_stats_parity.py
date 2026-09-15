"""Both arms' norm stats must agree on every shared feature.

They train on the identical merged root, so state and action statistics are the
same numbers. A difference means the data path diverged -- different root,
different filter, different transform order -- and it biases dQ in a way nothing
downstream catches, because each arm looks healthy on its own.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CMP = REPO / "scripts" / "compare_norm_stats.py"


def _stats(tmp_path: Path, name: str, mean, std, *, extra: dict | None = None) -> Path:
    d = {"state": {"mean": mean, "std": std}, "actions": {"mean": mean, "std": std}}
    if extra:
        d.update(extra)
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d))
    return p


def _run(a: Path, b: Path, *extra: str):
    return subprocess.run([sys.executable, str(CMP), "--arm-a", str(a), "--arm-b", str(b), *extra],
                          capture_output=True, text=True)


def test_identical_stats_pass(tmp_path):
    a = _stats(tmp_path, "a", [1.0, 2.0], [0.5, 0.25])
    b = _stats(tmp_path, "b", [1.0, 2.0], [0.5, 0.25])
    res = _run(a, b)
    assert res.returncode == 0, res.stdout
    assert "NORM_STATS_MATCH" in res.stdout
    assert "sha256" in res.stdout, "the fingerprint needs both hashes"


def test_a_shared_feature_that_differs_fails(tmp_path):
    """The case that silently biases dQ."""
    a = _stats(tmp_path, "a", [1.0, 2.0], [0.5, 0.25])
    b = _stats(tmp_path, "b", [1.0, 2.0000001], [0.5, 0.25])
    res = _run(a, b)
    assert res.returncode == 1, res.stdout
    assert "MISMATCH" in res.stdout and "state.mean" in res.stdout


def test_a_key_present_in_only_one_arm_is_tolerated(tmp_path):
    """Different feature sets are expected; different shared statistics are not."""
    a = _stats(tmp_path, "a", [1.0], [0.5])
    b = _stats(tmp_path, "b", [1.0], [0.5], extra={"progress": {"mean": [0.3], "std": [0.1]}})
    res = _run(a, b)
    assert res.returncode == 0, res.stdout
    assert "only in B" in res.stdout
    assert "NORM_STATS_MATCH" in res.stdout


def test_default_tolerance_is_exact(tmp_path):
    """Same data, same code, same numbers -- there is no reason to allow drift."""
    text = CMP.read_text()
    assert '"--tolerance", type=float, default=0.0' in text


def test_the_root_aware_computer_refuses_a_stale_path():
    """openpi's own compute_norm_stats takes only --config-name, and pi05_b1k ships
    dataset_root='./data/b1k/turning_on_radio'. Pointed at a slice elsewhere it
    would compute 1.25M frames of statistics over the wrong data and write them
    where training reads them."""
    text = (REPO / "scripts" / "compute_norm_stats_b1k.py").read_text()
    assert "does not exist -- refusing to compute stats over a stale path" in text
    assert "config resolved to" in text, "must also catch a root that silently did not take"
    assert "create_b1k_dataset" in text, "must use the root-aware loader"


def test_remove_strings_survives_pickling_for_spawned_workers():
    """num_workers > 0 means spawn, and spawn pickles every transform. A class
    defined inside main() cannot be pickled; the real run died on exactly that."""
    import importlib
    import pickle
    # Import it by a real module name that pickle can resolve again, as a
    # spawned worker would. (Loading it under an unregistered ad-hoc name made
    # this test fail for a reason that has nothing to do with the script.)
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        mod = importlib.import_module("compute_norm_stats_b1k")
    finally:
        sys.path.pop(0)
    restored = pickle.loads(pickle.dumps(mod.RemoveStrings()))
    assert restored({"a": 1.0, "prompt": "turn on"}) == {"a": 1.0}


def test_decode_free_mode_is_proven_before_it_is_used():
    """Skipping video decode is only safe if it changes nothing. The rung must
    prove bit-identical stats on seeded frames before the real pass uses it."""
    rung = (REPO / "scripts" / "session_b" / "rung1.sh").read_text()
    proof = rung.index("ns_equivalence.log")
    real = rung.index("3b. The real pass")
    assert proof < real, "the equivalence proof must run before the decode-free full pass"
    assert "cannot skip decoding" in rung


def test_decode_patch_is_applied_at_import_so_spawned_workers_get_it():
    """spawn re-imports the module in each worker but does not carry a runtime
    monkeypatch from the parent. The patch must live at module level behind an
    inherited env var."""
    text = (REPO / "scripts" / "compute_norm_stats_b1k.py").read_text()
    patch = text.index('os.environ.get("NORM_STATS_NO_DECODE") == "1"')
    assert patch < text.index("def main()"), "patch must be at module level, not inside main()"
