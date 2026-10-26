"""
app.py — Streamlit dashboard for Rubicon.

Pick a planted-leakage demo case or upload your own CSV, run the LangGraph
pipeline, and watch the supervisor interrogate the score before the models
are trusted. Launch with:  streamlit run app.py
"""
import io
import json
import os
import zipfile

import pandas as pd
import streamlit as st

from rubicon.graph import run_pipeline
from rubicon.testbed import CASES, ID_COL, TARGET, TREATMENT, make_case

st.set_page_config(page_title="Rubicon", layout="wide")
st.title("Rubicon — the AutoML that tries to break its own score")
st.caption("LangGraph · leakage probes · Optuna · MLflow · T-learner uplift · MCP")

with st.sidebar:
    st.header("Input")
    source = st.radio("Dataset", ["Planted demo case", "Upload CSV"])
    if source == "Planted demo case":
        case = st.selectbox("Leakage to plant", CASES, index=1)
        target, treatment, id_col = TARGET, TREATMENT, ID_COL
    else:
        upload = st.file_uploader("CSV file", type="csv")
        target = st.text_input("Target column", "churn")
        treatment = st.text_input("Treatment column (optional)", "")
        id_col = st.text_input("Id column (optional)", "")
    n_trials = st.slider("Optuna trials", 5, 50, 15)
    compare = st.checkbox("Run naive tournament first (before/after)", True)
    run = st.button("Run pipeline", type="primary", width="stretch")

if run:
    os.makedirs("data", exist_ok=True)
    ite = None
    if source == "Planted demo case":
        df, truth, ite = make_case(case, n=1500, seed=0)
        raw = f"data/demo_{case}.csv"
        df.to_csv(raw, index=False)
        st.info(f"Planted: **{truth['type']}** — correct fix: {truth['fix'] or 'nothing (control case)'}")
    else:
        if upload is None:
            st.stop()
        raw = "data/upload.csv"
        open(raw, "wb").write(upload.getbuffer())

    with st.status("Running LangGraph pipeline…", expanded=True) as status:
        try:
            s = run_pipeline(raw, target, treatment or None, id_col or None, ite,
                             compare_before_after=compare, n_trials=n_trials)
        except (ValueError, RuntimeError) as e:
            status.update(label="Pipeline stopped", state="error")
            st.error(str(e))
            st.stop()
        for line in s["logs"]:
            st.write(line)
        status.update(label="Done", state="complete")

    st.subheader("Audit")
    st.write(s["explanation"]["text"])
    st.caption(f"LLM mode: {s['llm']['mode']} · provider: {s['llm']['provider']} · "
               f"tokens: {s['llm']['usage']}")
    for r in s["rounds"]:
        with st.expander(f"Round {r['round']}" + (f" → {r['action']}" if r.get("action") else " → all clear")):
            st.dataframe(pd.DataFrame(r["probes"])[["probe", "score", "threshold", "flagged", "evidence"]],
                         width="stretch", hide_index=True)

    st.subheader("Model tournament")
    after = pd.DataFrame(s["tournament"]["leaderboard"]).T.rename(columns={"auc": "AUC after audit", "f1": "F1 after audit"})
    if "baseline" in s:
        before = pd.DataFrame(s["baseline"]["leaderboard"]).T.rename(columns={"auc": "AUC naive", "f1": "F1 naive"})
        table = before.join(after)
        table["inflation (AUC)"] = (table["AUC naive"] - table["AUC after audit"]).round(3)
        st.bar_chart(table[["AUC naive", "AUC after audit"]])
    else:
        table = after
    st.dataframe(table, width="stretch")
    t = s["tournament"]["tuned"]
    st.write(f"Optuna tuned **{t['family']}** → AUC {t['auc']} under *{s['plan']['cv']}* evaluation")

    if "causal" in s:
        st.subheader("Who is worth a discount?")
        c = s["causal"]
        cols = st.columns(3)
        if c.get("qini_normalised") is not None:
            cols[0].metric("Normalised Qini (1 = oracle)", c["qini_normalised"])
        else:
            cols[0].metric("Held-out Qini gain", c["qini_gain"])
        if "budget" in c:
            cols[1].metric("Budget lift vs random", f"{c['budget']['lift_vs_random_pct']}%")
            cols[2].metric("Share of oracle", f"{c['budget']['share_of_oracle_pct']}%")
        curves = pd.DataFrame({k: v for k, v in c["curves"].items() if k != "ks"}, index=c["curves"]["ks"])
        st.line_chart(curves)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(s["clean_csv"], "repaired_data.csv")
        zf.writestr("report.json", json.dumps({k: v for k, v in s.items()
                                               if k in ("plan", "rounds", "tournament", "baseline", "causal",
                                                        "explanation", "llm", "logs")}, indent=2, default=str))
    st.download_button("Download repaired data + report", buf.getvalue(), "rubicon_bundle.zip")
