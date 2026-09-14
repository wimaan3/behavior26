#!/usr/bin/env python3
"""Assert two arms' norm stats agree on every SHARED feature.

Both arms train on the identical merged root, so their state and action
statistics must be the same numbers. A difference means something diverged in the
data path -- a different root, a different filter, a different transform order --
and it would bias dQ in a way nothing downstream catches, because each arm's
training would look perfectly healthy on its own.

Upstream computes stats for ["state", "actions"] only; `progress` is not
normalised. So in practice the two files should be IDENTICAL. A key present in
one and absent in the other is reported and tolerated (that is the documented
"different feature sets" case); a shared key whose numbers differ is a failure.

Prints the sha256 of each file for the run fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = [float(x) for x in v]
        else:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="0.0 means bit-identical, which is what the same data on the "
                         "same code should produce.")
    a = ap.parse_args()

    A, B = json.load(open(a.arm_a)), json.load(open(a.arm_b))
    fa, fb = flatten(A), flatten(B)
    sa, sb = sha256(a.arm_a), sha256(a.arm_b)
    print(f"arm A sha256 {sa}\n      {a.arm_a}")
    print(f"arm B sha256 {sb}\n      {a.arm_b}")
    print(f"identical file: {sa == sb}")

    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    shared = sorted(set(fa) & set(fb))
    if only_a or only_b:
        print(f"keys only in A: {only_a}\nkeys only in B: {only_b}  (tolerated: different feature sets)")
    if not shared:
        print("FAIL: the two files share no keys at all")
        return 1

    bad = []
    for k in shared:
        va, vb = fa[k], fb[k]
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                bad.append((k, "length", len(va), len(vb))); continue
            for i, (x, y) in enumerate(zip(va, vb)):
                if (x != y) if a.tolerance == 0 else (abs(x - y) > a.tolerance):
                    bad.append((k, f"[{i}]", x, y)); break
        elif va != vb:
            bad.append((k, "", va, vb))

    print(f"shared keys: {len(shared)}  mismatched: {len(bad)}")
    for k, where, x, y in bad[:10]:
        print(f"  MISMATCH {k}{where}: A={x} B={y}")
    if bad:
        print("FAIL: both arms train on the same merged root, so shared statistics must match. "
              "A difference here biases dQ and nothing downstream catches it.")
        return 1
    print("NORM_STATS_MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
