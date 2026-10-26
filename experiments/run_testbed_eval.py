"""Experiment 1 — can the supervisor name the bug? 6 cases x 10 seeds.

For each planted leakage type we record whether the diagnosis contains the
planted type (detected), whether it contains ONLY the planted type (exact), and
whether anything else fired (false alarm). The clean case measures pure false
alarms. Writes results/detection.json.
"""
import json
import pathlib
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rubicon.supervisor import audit  # noqa: E402
from rubicon.testbed import CASES, TARGET, make_case  # noqa: E402

SEEDS = range(10)
N = 1500


def main():
    out, t0 = {}, time.time()
    for case in CASES:
        rows, conf = [], Counter()
        for seed in SEEDS:
            df, truth, _ = make_case(case, n=N, seed=seed)
            res = audit(df, TARGET)
            diag = set(res["diagnoses"])
            conf.update(diag)
            rows.append({"seed": seed, "diagnoses": sorted(diag), "rounds": len(res["rounds"]),
                         "final_plan": res["plan"]["cv"],
                         "round1_scores": {p["probe"]: p["score"] for p in res["rounds"][0]["probes"]}})
        planted = truth["type"]
        n = len(rows)
        out[case] = {
            "planted": planted, "n_seeds": n,
            "detected": None if planted == "clean" else sum(planted in r["diagnoses"] for r in rows) / n,
            "exact": None if planted == "clean" else sum(r["diagnoses"] == [planted] for r in rows) / n,
            "false_alarm": sum(bool(set(r["diagnoses"]) - {planted}) for r in rows) / n,
            "mean_rounds": sum(r["rounds"] for r in rows) / n,
            "diagnosis_counts": dict(conf), "runs": rows,
        }
        print(f"{case:17s} detected={out[case]['detected']} exact={out[case]['exact']} "
              f"false_alarm={out[case]['false_alarm']} rounds={out[case]['mean_rounds']:.1f}")
    out["_meta"] = {"n_rows": N, "seeds": list(SEEDS)}
    print(f"elapsed {time.time() - t0:.1f}s")
    (ROOT / "results" / "detection.json").write_text(json.dumps(out, indent=2))
    print("saved results/detection.json")


if __name__ == "__main__":
    main()
