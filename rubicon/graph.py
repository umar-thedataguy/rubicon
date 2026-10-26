"""
graph.py — the LangGraph workflow that ties the agents together.

  profile -> clean -> baseline -> probe --(flagged)--> repair --> probe ...
                                   |
                               (all clear)
                                   v
                              tournament -> causal -> explain -> END

`probe`/`repair` is a real conditional loop, bounded by max_rounds. `baseline`
runs the tournament under the naive plan BEFORE any repair so the final report
can show how much each model's score was inflated.
"""
import os
from typing import List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from .causal import evaluate_uplift
from .llm import explain_findings, generate_cleaning_script
from .supervisor import DEFAULT_PLAN, repair, run_probes
from .tools import profile, run_in_sandbox
from .tournament import run_tournament


class State(TypedDict, total=False):
    raw_csv: str
    clean_csv: str
    profile: dict
    target: str
    treatment: str
    id_col: str
    ite_table: object
    compare_before_after: bool
    n_trials: int
    max_rounds: int
    round: int
    plan: dict
    probe_results: list
    rounds: list
    llm: dict
    logs: List[str]
    baseline: dict
    tournament: dict
    causal: dict
    explanation: dict


def _log(state, msg):
    state.setdefault("logs", []).append(msg)


def profile_node(state):
    state["profile"] = profile(state["raw_csv"], state["target"])
    _log(state, f"[profiler] schema profile built: {state['profile']['rows']} rows, "
                f"{len(state['profile']['columns'])} columns, ~{state['profile']['approx_tokens']} tokens")
    return state


def clean_node(state):
    gen = generate_cleaning_script(state["profile"], state["raw_csv"], state["clean_csv"],
                                   state["target"], [])
    res = run_in_sandbox(gen["code"])
    ok = res["success"] and os.path.exists(state["clean_csv"])
    if ok:
        df = pd.read_csv(state["clean_csv"])
        ok = state["target"] in df.columns and len(df) > 0
    if not ok and gen["mode"] == "llm":
        _log(state, f"[cleaner] LLM script failed in sandbox ({res['stderr'][:120]}); using fallback")
        from .llm import fallback_cleaning_script
        res = run_in_sandbox(fallback_cleaning_script(state["raw_csv"], state["clean_csv"],
                                                      state["target"], []))
        gen["mode"] = "llm-failed-fallback"
    if not res["success"]:
        raise RuntimeError(f"cleaning failed: {res['stderr']}")
    state["llm"] = {"mode": gen["mode"], "provider": gen["provider"], "model": gen["model"],
                    "usage": gen["usage"]}
    _log(state, f"[cleaner] mode={gen['mode']} provider={gen['provider']} "
                f"tokens={gen['usage']} sandbox -> {res['stdout']}")
    state["plan"] = {**DEFAULT_PLAN, "dropped_columns": [], "resolved": []}
    state["round"], state["rounds"] = 0, []
    return state


def baseline_node(state):
    if not state.get("compare_before_after"):
        return state
    df = pd.read_csv(state["clean_csv"])
    state["baseline"] = run_tournament(df, state["target"], state["plan"],
                                       n_trials=state.get("n_trials", 20), run_name="naive")
    _log(state, "[baseline] naive tournament (before any audit): " + ", ".join(
        f"{k} {v['auc']:.3f}" for k, v in state["baseline"]["leaderboard"].items()))
    return state


def probe_node(state):
    df = pd.read_csv(state["clean_csv"])
    state["round"] += 1
    results = run_probes(df, state["target"], state["plan"])
    state["probe_results"] = results
    state["rounds"].append({"round": state["round"], "probes": results})
    flagged = [r["probe"] for r in results if r["flagged"]]
    _log(state, f"[supervisor] round {state['round']}: " +
         (f"flagged {flagged} — " + results[[r['probe'] for r in results].index(flagged[0])]["evidence"]
          if flagged else "all probes clear"))
    return state


def route(state):
    flagged = any(r["flagged"] for r in state["probe_results"])
    return "repair" if flagged and state["round"] < state.get("max_rounds", 5) else "tournament"


def repair_node(state):
    df = pd.read_csv(state["clean_csv"])
    finding = next(r for r in state["probe_results"] if r["flagged"])
    df, plan, desc = repair(df, finding, state["plan"], state["target"])
    df.to_csv(state["clean_csv"], index=False)
    state["plan"] = plan
    state["rounds"][-1]["action"] = desc
    _log(state, f"[repair] {desc}")
    return state


def tournament_node(state):
    df = pd.read_csv(state["clean_csv"])
    state["tournament"] = run_tournament(df, state["target"], state["plan"],
                                         n_trials=state.get("n_trials", 20), run_name="repaired")
    t = state["tournament"]
    _log(state, f"[tournament] plan={t['plan']}: " + ", ".join(
        f"{k} {v['auc']:.3f}" for k, v in t["leaderboard"].items()) +
        f" | tuned {t['tuned']['family']} {t['tuned']['auc']:.3f}")
    return state


def causal_node(state):
    df = pd.read_csv(state["clean_csv"])
    tr = state.get("treatment")
    if tr and tr in df.columns:
        state["causal"] = evaluate_uplift(df, state["target"], tr, state.get("ite_table"),
                                          state.get("id_col"), plan=state["plan"])
        c = state["causal"]
        _log(state, f"[causal] held-out Qini gain {c['qini_gain']}" +
             (f", normalised {c.get('qini_normalised')}, budget lift vs random "
              f"{c['budget']['lift_vs_random_pct']}%" if "budget" in c else ""))
    return state


def explain_node(state):
    state["explanation"] = explain_findings(state["rounds"], state["plan"])
    _log(state, f"[explain] ({state['explanation']['mode']}) {state['explanation']['text']}")
    return state


def build_graph():
    g = StateGraph(State)
    for name, fn in [("profile", profile_node), ("clean", clean_node), ("baseline", baseline_node),
                     ("probe", probe_node), ("repair", repair_node),
                     ("tournament", tournament_node), ("causal", causal_node),
                     ("explain", explain_node)]:
        g.add_node(name, fn)
    g.set_entry_point("profile")
    g.add_edge("profile", "clean")
    g.add_edge("clean", "baseline")
    g.add_edge("baseline", "probe")
    g.add_conditional_edges("probe", route, {"repair": "repair", "tournament": "tournament"})
    g.add_edge("repair", "probe")
    g.add_edge("tournament", "causal")
    g.add_edge("causal", "explain")
    g.add_edge("explain", END)
    return g.compile()


def run_pipeline(raw_csv, target="churn", treatment=None, id_col=None, ite_table=None,
                 compare_before_after=True, n_trials=20, max_rounds=5, clean_csv=None):
    state = {"raw_csv": raw_csv, "clean_csv": clean_csv or raw_csv.replace(".csv", "_clean.csv"),
             "target": target, "treatment": treatment, "id_col": id_col, "ite_table": ite_table,
             "compare_before_after": compare_before_after, "n_trials": n_trials,
             "max_rounds": max_rounds, "logs": []}
    return build_graph().invoke(state)
