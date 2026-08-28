"""A/B/C census of the 100 BEHAVIOR-2026 challenge goal definitions.

Answers one question: at episode end, how much of the scored goal state is
actually visible in a single RGB-D observation? That decides whether the
predicate head needs memory.

Every grounded goal literal is bucketed:

    A  resolvable from the final RGB-D frame alone (geometry or appearance,
       normally in view)
    B  occludable -- a real visual signature exists, but the thing being
       scored can be hidden at episode end (inside a closed container, under
       furniture). Needs memory or active looking.
    C  no discriminative visual signature even with a perfect view.

Grounding is done with BDDL's own compiler on the same code path the scorer
uses, so the literal counts here are the actual Q denominators.

Usage:
    python -m analysis.goal_census --bddl <path/to/BEHAVIOR-1K/bddl3> \
        --tasks <path/to/challenge_tasks.json> --out-dir <dir>

challenge_tasks.json is scraped by scripts/fetch_challenge_tasks.py.
Requires the bddl3 package importable (deps: future, networkx, nltk, numpy).
It does NOT require OmniGibson or a GPU.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Verified against BEHAVIOR-1K v3.9.2:
#
#   OmniGibson/omnigibson/metrics/task_metric.py::TaskMetric._compute_episode_metrics
#       q_score.final = 1.0 if task.success else
#           max over ground_goal_state_options of
#               (# literals TRUE now that were FALSE at reset) / len(option)
#
#   Two consequences the rest of this file is built around:
#     1. The denominator is the number of GROUNDED literals in an option, not
#        the number of surface conjuncts. forall over 3 cans = 3 literals.
#     2. A goal literal already satisfied at reset contributes 0 to the
#        numerator but still counts in the denominator. Initially-true
#        literals are dead weight for partial credit.
#
#   OmniGibson/omnigibson/utils/bddl_utils.py::evaluate_bddl_predicate
#       real(x)   := x is not None   (object exists -- transition-rule product)
#       future(x) := x is None
# ---------------------------------------------------------------------------

# Per-predicate bucket. Entries that need argument types are resolved in
# classify() instead; this table holds only the unconditional cases.
#
# Evidence for the non-obvious calls, all from BEHAVIOR-1K v3.9.2:
#   toggled_on -> A: object_states/toggle.py renders a visual marker that is
#       green when on and red when off, and sets visible = True unconditionally
#       (toggle.py:127, toggle.py:171). Directly legible in RGB.
#   cooked     -> B: object_states/cooked.py defines get_texture_change_params
#       (brown tint), so there IS a signature -- but cooked objects usually end
#       up inside an oven/microwave/pot, hence occludable rather than A.
#   frozen     -> B: object_states/frozen.py get_texture_change_params (white
#       tint), same occlusion argument (freezer).
#   touching / attached -> C: these are PhysX contact / AttachedTo joint facts.
#       Contact versus a 2 mm gap is not resolvable at 720x720, and an attached
#       poster looks identical to one merely resting against the nail.
#       JUDGMENT CALL: they are geometric, not invisible. If you would rather
#       treat "sub-resolution but geometric" as B, move them and re-run.
UNCONDITIONAL = {
    "ontop": "A",        # support relation, visible
    "nextto": "A",       # horizontal adjacency, visible
    "open": "A",         # joint angle of a door/drawer, visible
    "toggled_on": "A",   # rendered toggle marker, green/red
    "on_fire": "A",      # flames
    "covered": "A",      # particle system rendered on the surface
    "under": "B",        # occluded by construction
    "cooked": "B",
    "frozen": "B",
    "filled": "B",       # must see into the container
    "contains": "B",     # must see into the container
    "touching": "C",
    "attached": "C",
}


def synset_of(instance):
    """apple.n.01_1 -> apple.n.01 ; apple.n.01_* -> apple.n.01"""
    head, _, tail = instance.rpartition("_")
    return head if (tail.isdigit() or tail == "*") else instance


def classify(negated, pred, args, props):
    """Return (bucket, reason) for one grounded goal literal.

    props: {synset -> set of property names} from bddl generated_data.
    """
    def prop(inst):
        return props.get(synset_of(inst), set())

    if pred == "inside":
        # Visible only if you can see into the container without opening it.
        container = args[1]
        if "openable" in prop(container):
            return "B", "inside a closed-able container (%s)" % synset_of(container)
        return "A", "inside an open container (%s)" % synset_of(container)

    if pred == "real":
        # real(x) is the product of a transition rule firing.
        if "substance" in prop(args[0]):
            # e.g. cooked__popcorn appearing inside a bag -- nothing to see.
            return "C", "substance came into existence (%s)" % synset_of(args[0])
        if negated:
            # Object was destroyed. Absence of evidence is not evidence of
            # absence: an occluded object looks the same as a consumed one.
            return "B", "object consumed (%s)" % synset_of(args[0])
        return "A", "new rigid object appeared (%s)" % synset_of(args[0])

    bucket = UNCONDITIONAL.get(pred)
    if bucket is None:
        return "?", "UNCLASSIFIED predicate %r -- extend UNCONDITIONAL" % pred
    return bucket, pred


def ground_options(conds, get_object_scope, get_goal_conditions):
    """Enumerate grounded goal options exactly as bddl's own scorer does.

    Mirrors bddl.condition_evaluation.get_ground_state_options: cartesian
    product of each top-level conjunct's flattened_condition_options, drop
    self-contradictory combinations, shortest first. We re-implement the
    product here only so we keep the raw token lists (the compiled HEAD
    objects do not expose the literal text).
    """
    scope = get_object_scope(conds)
    goal_conditions = get_goal_conditions(conds, scope, generate_ground_options=True)
    raw = [
        list(itertools.chain(*combo))
        for combo in itertools.product(
            *[c.flattened_condition_options for c in goal_conditions]
        )
    ]
    consistent = []
    for option in raw:
        if any(
            (a[0] == "not" and a[1] == b) or (b[0] == "not" and b[1] == a)
            for a, b in itertools.combinations(option, 2)
        ):
            continue
        consistent.append(option)
    consistent.sort(key=len)
    return consistent


def denominator_varies(task, conds_factory, ground_fn, extra=3):
    """Does the Q denominator depend on how many objects the scene contains?

    Instance names ending in _* are wildcards: bddl/wildcard.py expands them
    against the actual scene layout ("as many cabinets as this kitchen has").
    Goals may not contain wildcards directly, but a `forall` over a wildcarded
    synset still picks them up, so the grounded literal count -- the Q
    denominator -- can differ per instance.

    Detected by re-grounding with each wildcard replaced by `extra` concrete
    instances and checking whether the option size moved.

    Returns (varies: bool, wildcard_synsets: list, base_sizes, expanded_sizes).
    """
    conds = conds_factory(task)
    wildcarded = sorted(
        {s for s, insts in conds.parsed_objects.items()
         if any(i.endswith("_*") for i in insts)}
    )
    base_sizes = sorted({len(o) for o in ground_fn(conds)})
    if not wildcarded:
        return False, [], base_sizes, base_sizes

    expanded = conds_factory(task)
    for synset in wildcarded:
        concrete = [i for i in expanded.parsed_objects[synset]
                    if not i.endswith("_*")]
        expanded.parsed_objects[synset] = concrete + [
            "%s_%d" % (synset, len(concrete) + k + 1) for k in range(extra)
        ]
    exp_sizes = sorted({len(o) for o in ground_fn(expanded)})
    return base_sizes != exp_sizes, wildcarded, base_sizes, exp_sizes


def as_literal(tokens):
    """['not', ['inside', 'a', 'b']] -> (True, 'inside', ('a', 'b'))"""
    if tokens[0] == "not":
        return True, tokens[1][0], tuple(tokens[1][1:])
    return False, tokens[0], tuple(tokens[1:])


# Unary states OmniGibson initialises to False unless the BDDL :init says
# otherwise, so a goal literal asserting the negative starts out satisfied.
CLOSED_WORLD_FALSE = {"open", "toggled_on", "cooked", "frozen", "on_fire",
                      "covered", "filled", "contains", "attached", "saturated"}
KINEMATIC = {"ontop", "inside", "nextto", "under", "touching"}


def initially_true(negated, pred, args, init_set, future_set):
    """Estimate whether a goal literal already holds at reset.

    Returns True / False / None (unknown). This is an ESTIMATE from the BDDL
    :init block under closed-world assumptions -- the scorer evaluates it in
    the live simulator at reset. Verify against a real rollout before relying
    on any single task's number.
    """
    if pred == "real":
        # future(x) in :init means x does not exist yet.
        exists_at_init = args[0] not in future_set
        return (not exists_at_init) if negated else exists_at_init

    stated_positive = (pred, args) in init_set
    stated_negative = ("not", pred, args) in init_set

    if negated:
        if stated_negative:
            return True
        if stated_positive:
            return False
        if pred in CLOSED_WORLD_FALSE:
            return True   # defaults false, so "not P" holds at reset
        if pred in KINEMATIC:
            return None   # cannot tell without the sampled scene
        return None

    if stated_positive:
        return True
    if stated_negative:
        return False
    if pred in CLOSED_WORLD_FALSE:
        return False
    if pred in KINEMATIC:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bddl", required=True,
                    help="Path to the bddl3 package root (BEHAVIOR-1K/bddl3)")
    ap.add_argument("--tasks", required=True,
                    help="challenge_tasks.json from scripts/fetch_challenge_tasks.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.bddl)
    from bddl.activity import Conditions, get_object_scope, get_goal_conditions

    props_path = os.path.join(args.bddl, "bddl", "generated_data",
                              "propagated_annots_canonical.json")
    with open(props_path, encoding="utf-8") as f:
        props = {k: set(v) for k, v in json.load(f).items()}

    with open(args.tasks, encoding="utf-8") as f:
        tasks = [t["id"] for t in json.load(f)]
    os.makedirs(args.out_dir, exist_ok=True)

    rows, detail = [], {}
    unclassified = Counter()
    bucket_totals = Counter()
    per_predicate = defaultdict(Counter)

    def conds_factory(name):
        return Conditions(name, 0, "behavior-1k")

    def ground_fn(c):
        return ground_options(c, get_object_scope, get_goal_conditions)

    for task in tasks:
        conds = conds_factory(task)
        options = ground_fn(conds)
        sizes = sorted({len(o) for o in options})

        # Does the number of grounded literals depend on how many wildcard
        # objects the scene happens to contain? If so the Q denominator is
        # instance-dependent and the count below is only the minimum.
        varies, wildcard_synsets, _, exp_sizes = denominator_varies(
            task, conds_factory, ground_fn)

        init_set, future_set = set(), set()
        for tokens in conds.parsed_initial_conditions:
            neg, pred, a = as_literal(tokens)
            if pred == "future":
                future_set.add(a[0])
            init_set.add(("not", pred, a) if neg else (pred, a))

        literals = [as_literal(t) for t in options[0]]
        buckets, reasons, init_flags = [], [], []
        for neg, pred, a in literals:
            bucket, reason = classify(neg, pred, a, props)
            if bucket == "?":
                unclassified[pred] += 1
            buckets.append(bucket)
            reasons.append(reason)
            init_flags.append(initially_true(neg, pred, a, init_set, future_set))
            bucket_totals[bucket] += 1
            per_predicate[("not " if neg else "") + pred][bucket] += 1

        n = len(literals)
        n_init_true = sum(1 for f in init_flags if f is True)
        rows.append({
            "task": task,
            "n_goal_literals": n,
            "n_A": buckets.count("A"),
            "n_B": buckets.count("B"),
            "n_C": buckets.count("C"),
            "frac_A": round(buckets.count("A") / n, 4),
            "n_ground_options": len(options),
            "option_sizes": "|".join(str(s) for s in sizes),
            "denominator_varies_with_scene": varies,
            "denominator_if_3_extra": "|".join(str(s) for s in exp_sizes),
            "wildcard_synsets": "|".join(wildcard_synsets),
            "est_initially_true": n_init_true,
            "est_reachable_ceiling": round((n - n_init_true) / n, 4),
            "predicates": " ".join(
                ("!" if neg else "") + pred for neg, pred, _ in literals),
        })
        detail[task] = [
            {"literal": ("not " if neg else "") + pred + "(" + ", ".join(a) + ")",
             "bucket": b, "reason": r, "initially_true": f}
            for (neg, pred, a), b, r, f in zip(literals, buckets, reasons, init_flags)
        ]

    csv_path = os.path.join(args.out_dir, "goal_census.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "goal_census_detail.json"),
              "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=1)

    total = sum(bucket_totals.values())
    n_lits = sorted(r["n_goal_literals"] for r in rows)
    print("tasks parsed: %d   grounded goal literals: %d" % (len(rows), total))
    print("literals/task  mean %.2f  median %d  min %d  max %d"
          % (total / len(rows), n_lits[len(n_lits) // 2], n_lits[0], n_lits[-1]))
    print()
    for b in ("A", "B", "C", "?"):
        if bucket_totals[b]:
            print("  %s  %4d  %6.1f%%" % (b, bucket_totals[b],
                                          100 * bucket_totals[b] / total))
    if unclassified:
        print("\n  UNCLASSIFIED predicates: %s" % dict(unclassified))

    print("\nper-predicate:")
    for pred, c in sorted(per_predicate.items(), key=lambda kv: -sum(kv[1].values())):
        print("  %5d  %-18s %s" % (sum(c.values()), pred, dict(c)))

    all_a = [r for r in rows if r["n_A"] == r["n_goal_literals"]]
    no_a = [r for r in rows if r["n_A"] == 0]
    varies = [r["task"] for r in rows if r["denominator_varies_with_scene"]]
    print("\ntasks fully A (memory-free scoring possible):   %d" % len(all_a))
    print("tasks with NO A literal (blind without memory): %d" % len(no_a))
    for r in no_a:
        print("    %-44s %s" % (r["task"], r["predicates"]))
    print("tasks whose denominator varies with the scene: %d %s" % (len(varies), varies))
    print("\nwrote %s" % csv_path)


if __name__ == "__main__":
    main()
