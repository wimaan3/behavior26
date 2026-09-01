#!/usr/bin/env python3
"""Rank BEHAVIOR-1K tasks as training candidates and audit their reward instrumentation.

Inputs (this directory)      : behavior1k_task_table.csv, behavior1k_episode_stats.csv
Inputs (../census, committed): goal_census_detail.json  -- per-literal BDDL goal facts
Output                       : task_shortlist.csv

Key identity, verified on all 4,723 valid episodes:
    phi0 * D  is an integer  ==  number of goal units that NEVER fired
    D * (1 - phi0)           ==  number of reward events actually observed
"""
import csv, json, os, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, os.pardir, "census", "goal_census_detail.json")

tt = {r["task_name"]: r for r in csv.DictReader(
    open(os.path.join(HERE, "behavior1k_task_table.csv")))}
eps = list(csv.DictReader(open(os.path.join(HERE, "behavior1k_episode_stats.csv"))))
census = json.load(open(CENSUS))

lens = collections.defaultdict(list)
for e in eps:
    lens[e["task_name"]].append(float(e["T"]))
meanT = {k: statistics.mean(v) for k, v in lens.items()}
CORPUS_MEAN = statistics.mean(meanT.values())


def audit(name):
    """Return (must_flip, n_literals, reward_events, denominator_status)."""
    D = int(tt[name]["D"])
    lits = census[name]
    # initially_true is True / False / None(unknown); count anything not proven
    # already-satisfied as a predicate that must flip.
    must_flip = sum(1 for l in lits if l["initially_true"] is not True)
    events = D * (1.0 - float(tt[name]["phi0_mean"]))
    if D < must_flip:
        status = "COLLAPSED"      # denominator smaller than the goal it represents
    elif events < D - 1e-6:
        status = "UNDER_FIRES"    # denominator right, but units never credited
    else:
        status = "CONSISTENT"
    return must_flip, len(lits), events, status


def row(name, tier, rank):
    r = tt[name]
    D, T = int(r["D"]), meanT[name]
    mf, nl, ev, status = audit(name)
    return {
        "rank": rank,
        "tier": tier,
        "task": name,
        "D": D,
        "mean_episode_len": round(T),
        "eval_timeout_frames": round(1.5 * T),      # organizers: 1.5x own mean demo
        "rel_eval_cost": round(T / CORPUS_MEAN, 2),  # 1.00 == corpus-average task
        "valid_rate": round(float(r["valid_rate"]), 3),
        "episodes": int(r["episodes"]),
        "n_valid": int(r["n_valid"]),
        "phi0_mean": round(float(r["phi0_mean"]), 4),
        "must_flip_predicates": mf,
        "n_goal_literals": nl,
        "reward_events_observed": round(ev, 2),
        "denominator_status": status,
        "reward_instrumentation": r["reward_instrumentation"],
    }


# ---- Tier 1: phi0 == 0 in every episode, valid_rate >= 70%, ranked by length ----
primary = [n for n, r in tt.items()
           if r["D"] and float(r["phi0_frac_gt0"]) == 0.0
           and float(r["valid_rate"]) >= 0.70]
primary.sort(key=lambda n: meanT[n])

# ---- Tier 2: deliberate medium-D picks (real progress structure) ----
MEDIUM = ["cook_bacon", "make_rose_centerpieces", "chop_an_onion"]

rows = [row(n, "primary", i + 1) for i, n in enumerate(primary[:12])]
rows += [row(n, "medium_D", None) for n in MEDIUM if n not in primary[:12]]

out = os.path.join(HERE, "task_shortlist.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote %s (%d rows); corpus mean episode length = %.0f" % (out, len(rows), CORPUS_MEAN))
