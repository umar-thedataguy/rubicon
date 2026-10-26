"""
causal.py — from "who will churn" to "who is worth a discount".

A T-learner (one model for treated customers, one for untreated) estimates each
customer's Individual Treatment Effect: how much the discount lowers THEIR churn
probability. Everything is scored on a held-out 30%:

  * Qini curve / normalised Qini coefficient  (the standard uplift metric)
  * correlation with the TRUE effect, when the testbed's truth table is given
  * a budget simulation that scores a targeting policy with the TRUE effect,
    against random targeting and an oracle that knows the truth

The ROI number is therefore a measurement, not an identity: a bad ITE model
would lose to random targeting here.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .probes import encode, id_like

SEED = 42


def qini(y_retained, treated, score, n_points=50):
    """Qini curve values at n_points cut-offs when ranking by `score` (desc)."""
    order = np.argsort(-score, kind="stable")
    y, t = y_retained[order], treated[order]
    n = len(y)
    ks = np.linspace(0, n, n_points + 1).astype(int)[1:]
    curve = []
    for k in ks:
        yt, nt = y[:k][t[:k] == 1].sum(), (t[:k] == 1).sum()
        yc, nc = y[:k][t[:k] == 0].sum(), (t[:k] == 0).sum()
        curve.append(float(yt - yc * nt / nc) if nc > 0 else 0.0)
    curve = np.array(curve)
    random = curve[-1] * ks / n
    gain = float(np.trapezoid(curve - random, ks))
    return {"ks": ks.tolist(), "curve": curve.tolist(), "random": random.tolist(), "gain": gain}


def holdout_split(df, y, t, plan, seed):
    """70/30 split that honours the repaired evaluation plan: customers never
    straddle the split under group leakage; the test set is the FUTURE under drift."""
    idx = np.arange(len(df))
    if plan and plan.get("cv") == "group" and plan.get("group_col"):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        return next(gss.split(idx, groups=df[plan["group_col"]]))
    if plan and plan.get("cv") == "temporal" and plan.get("time_col"):
        order = np.argsort(df[plan["time_col"]].values, kind="stable")
        cut = int(0.7 * len(df))
        return order[:cut], order[cut:]
    return train_test_split(idx, test_size=0.3, random_state=seed, stratify=t * 2 + y)


def evaluate_uplift(df, target, treatment, ite_table=None, id_col=None, plan=None,
                    ltv=200.0, budget_frac=0.25, seed=SEED):
    X, y = encode(df, target)
    ids = df[id_col].values if id_col else None
    X = X.drop(columns=id_like(df), errors="ignore")
    t = X.pop(treatment).values
    idx_tr, idx_te = holdout_split(df, y.values, t, plan, seed)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte, ttr, tte = y.values[idx_tr], y.values[idx_te], t[idx_tr], t[idx_te]

    m1 = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr[ttr == 0], ytr[ttr == 0])
    ite_hat = m0.predict_proba(Xte)[:, 1] - m1.predict_proba(Xte)[:, 1]  # churn reduction

    retained = 1 - yte
    q_model = qini(retained, tte, ite_hat)
    out = {"n_test": int(len(idx_te)), "split": (plan or {}).get("cv", "stratified"),
           "qini_gain": round(q_model["gain"], 2),
           "mean_ite_hat": round(float(ite_hat.mean()), 4),
           "curves": {"ks": q_model["ks"], "model": q_model["curve"], "random": q_model["random"]}}

    if ite_table is not None and id_col:
        truth = pd.Series(ite_table.set_index(id_col)["true_ite"])
        true_te = truth.reindex(ids[idx_te]).values
        ok = ~np.isnan(true_te)
        q_oracle = qini(retained[ok], tte[ok], true_te[ok])
        out["spearman_vs_truth"] = round(float(spearmanr(ite_hat[ok], true_te[ok])[0]), 3)
        out["qini_normalised"] = round(q_model["gain"] / q_oracle["gain"], 3) if q_oracle["gain"] else None
        out["curves"]["oracle"] = q_oracle["curve"]
        # budget simulation, scored with the TRUE effect
        k = int(budget_frac * ok.sum())
        true_ok, hat_ok = true_te[ok], ite_hat[ok]
        policy = true_ok[np.argsort(-hat_ok)[:k]].sum() * ltv
        oracle = np.sort(true_ok)[::-1][:k].sum() * ltv
        random = true_ok.mean() * k * ltv
        out["budget"] = {"targeted_customers": k, "ltv_per_saved": ltv,
                         "saved_value_policy": round(float(policy), 1),
                         "saved_value_random": round(float(random), 1),
                         "saved_value_oracle": round(float(oracle), 1),
                         "lift_vs_random_pct": round(float(policy / random - 1) * 100, 1) if random > 0 else None,
                         "share_of_oracle_pct": round(float(policy / oracle) * 100, 1) if oracle > 0 else None}
    return out
