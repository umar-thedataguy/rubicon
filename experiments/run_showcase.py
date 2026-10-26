"""Experiment 2 — the full LangGraph pipeline, end to end, on three planted cases.

Each case runs the naive tournament BEFORE the audit and the repaired tournament
AFTER it, so we can see how much each model's score was inflated, then the
causal stage scored against the testbed's true treatment effect.
Writes results/showcase.json (and the clean/leaky CSVs under data/).
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rubicon.graph import run_pipeline  # noqa: E402
from rubicon.testbed import ID_COL, TARGET, TREATMENT, make_case  # noqa: E402

CASES = [("proxy_feature", 3000), ("group_leakage", 1200), ("temporal_drift", 3000)]


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    out = {}
    for case, n in CASES:
        t0 = time.time()
        df, truth, ite = make_case(case, n=n, seed=0)
        raw = ROOT / "data" / f"{case}.csv"
        df.to_csv(raw, index=False)
        s = run_pipeline(str(raw), TARGET, TREATMENT, ID_COL, ite, compare_before_after=True,
                         n_trials=15)
        before, after = s["baseline"]["leaderboard"], s["tournament"]["leaderboard"]
        out[case] = {
            "planted": truth, "rows": len(df),
            "llm": s["llm"], "diagnoses": [r.get("action") for r in s["rounds"] if r.get("action")],
            "rounds": s["rounds"], "plan": s["plan"],
            "leaderboard_before": before, "leaderboard_after": after,
            "inflation_auc": {m: round(before[m]["auc"] - after[m]["auc"], 4) for m in before},
            "tuned_before": s["baseline"]["tuned"], "tuned_after": s["tournament"]["tuned"],
            "causal": {k: v for k, v in s.get("causal", {}).items() if k != "curves"},
            "causal_curves": s.get("causal", {}).get("curves"),
            "explanation": s["explanation"], "logs": s["logs"],
        }
        print(f"\n=== {case} ({time.time() - t0:.1f}s)")
        for line in s["logs"]:
            print("  " + line)
        (ROOT / "results" / "showcase.json").write_text(json.dumps(out, indent=2, default=str))
    print("saved results/showcase.json")


if __name__ == "__main__":
    main()
