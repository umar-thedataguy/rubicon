"""
supervisor.py — turns probe results into a diagnosis and a repair.

The supervisor is deliberately boring: run the probes in a fixed order, act on
the first one that fires, re-probe, repeat (bounded). Two kinds of fix exist:
  * data fixes  — drop a column, drop rows        (the data changes)
  * plan fixes  — change HOW we evaluate           (the data stays; cv strategy changes)
"""
from .probes import PROBES, near_duplicate_mask

DEFAULT_PLAN = {"cv": "stratified", "group_col": None, "time_col": None,
                "dropped_columns": [], "rows_removed": 0, "resolved": []}


def run_probes(df, target, plan):
    """Run every probe; probes already handled by a plan fix cannot re-fire."""
    out = []
    for probe in PROBES:
        r = probe(df, target, plan=plan)
        if r["probe"] in plan["resolved"]:
            r["flagged"] = False
            r["evidence"] += " (already handled by evaluation plan)"
        out.append(r)
    return out


def repair(df, finding, plan, target):
    """Apply the fix for one finding. Returns (df, plan, description)."""
    plan = {**plan, "dropped_columns": list(plan["dropped_columns"]),
            "resolved": list(plan["resolved"])}
    kind = finding["probe"]
    if kind == "proxy_feature":
        df = df.drop(columns=[finding["column"]])
        plan["dropped_columns"].append(finding["column"])
        desc = f"dropped '{finding['column']}'"
    elif kind == "exact_duplicates":
        n0 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        plan["rows_removed"] += n0 - len(df)
        desc = f"removed {n0 - len(df)} duplicate rows"
    elif kind == "near_duplicates":
        _, drop = near_duplicate_mask(df, target)
        df = df.loc[~drop].reset_index(drop=True)
        plan["rows_removed"] += int(drop.sum())
        desc = f"removed {int(drop.sum())} near-duplicate rows"
    elif kind == "group_leakage":
        plan["cv"], plan["group_col"] = "group", finding["column"]
        plan["resolved"].append(kind)
        desc = f"switched to group-aware CV on '{finding['column']}'"
    elif kind == "temporal_drift":
        plan["cv"], plan["time_col"] = "temporal", finding["column"]
        plan["resolved"].append(kind)
        desc = f"switched to forward-chaining evaluation on '{finding['column']}'"
    else:
        desc = "no fix known"
    return df, plan, desc


def audit(df, target, max_rounds=5):
    """Full probe -> repair loop. Returns dict with final df, plan and a log of rounds."""
    plan = {**DEFAULT_PLAN, "dropped_columns": [], "resolved": []}
    rounds, diagnoses = [], []
    for rnd in range(1, max_rounds + 1):
        results = run_probes(df, target, plan)
        flagged = [r for r in results if r["flagged"]]
        rounds.append({"round": rnd, "probes": results})
        if not flagged:
            break
        df, plan, desc = repair(df, flagged[0], plan, target)
        diagnoses.append(flagged[0]["probe"])
        rounds[-1]["action"] = desc
    return {"df": df, "plan": plan, "rounds": rounds, "diagnoses": diagnoses}
