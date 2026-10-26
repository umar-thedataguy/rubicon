"""
probes.py — five cheap tests that interrogate a suspiciously good score.

Each probe returns the same shape:
  {"probe": name, "score": float, "threshold": float, "flagged": bool,
   "evidence": str, "column": str | None}

Probes never train the final models; they use a fast gradient-boosting model
only to compare evaluation schemes. Thresholds are deliberately simple and are
reported in the results so they can be criticised.
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42


def encode(df, target):
    """Numeric view of a frame: factorise text columns, median-fill gaps.

    The target is normalised to {0, 1}; a non-binary target is an error,
    because every probe compares binary-classification scores."""
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found; columns are {list(df.columns)}")
    X = df.drop(columns=[target]).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]
    X = X.fillna(X.median(numeric_only=True))
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    values = sorted(y.dropna().unique())
    if len(values) != 2:
        raise ValueError(f"target {target!r} must be binary; found {len(values)} distinct values")
    return X, (y == values[1]).astype(int)


def _tokens(name):
    """'customer_id' -> ['customer','id'];  'signupMonth' -> ['signup','month']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def id_like(df):
    """Columns that NAME an entity id (customer_id, orderId, ID) — never 'paid' or 'valid'."""
    return [c for c in df.columns if "id" in _tokens(c)]


def time_like(df):
    """Columns that index records in time (signup_month, event_date, timestamp) —
    not durations or rates like tenure_months or monthly_charges."""
    keys = {"date", "time", "timestamp", "datetime", "month", "year", "week", "day", "period"}
    return [c for c in df.columns if keys & set(_tokens(c))]


def _fast_model():
    return HistGradientBoostingClassifier(max_iter=100, random_state=SEED)


def _strat():
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def _reference_cv(df, plan):
    """The CV the pipeline currently trusts: group-aware once group leakage is handled."""
    if plan and plan.get("group_col"):
        return GroupKFold(5), df[plan["group_col"]]
    return _strat(), None


def probe_proxy(df, target, threshold=0.95, plan=None):
    """Does any SINGLE feature predict the target almost perfectly?"""
    X, y = encode(df, target)
    best_col, best = None, 0.5
    for c in X.columns:
        if X[c].nunique() < 2:
            continue
        auc = roc_auc_score(y, X[c])
        auc = max(auc, 1 - auc)
        if auc > best:
            best, best_col = auc, c
    return {"probe": "proxy_feature", "score": round(float(best), 3), "threshold": threshold,
            "flagged": best >= threshold, "column": best_col,
            "evidence": f"single-feature AUC of '{best_col}' = {best:.3f}"}


def probe_exact_duplicates(df, target, threshold=0.02, plan=None):
    """What fraction of rows are exact copies of another row?"""
    frac = float(df.duplicated().mean())
    return {"probe": "exact_duplicates", "score": round(frac, 3), "threshold": threshold,
            "flagged": frac >= threshold, "column": None,
            "evidence": f"{frac:.1%} of rows are exact duplicates"}


def near_duplicate_mask(df, target):
    """Rows whose nearest neighbour (in standardised feature space) is closer than
    10% of the typical nearest-neighbour distance. Returns (close, drop): `close`
    marks every row in a close pair, `drop` keeps one row of each pair."""
    X, _ = encode(df, target)
    X = X.drop(columns=id_like(df) + time_like(df), errors="ignore")
    Z = StandardScaler().fit_transform(X)
    ids = df[id_like(df)[0]].values if id_like(df) else np.arange(len(df))
    k = min(6, len(df))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
    dist[ids[idx] == ids[:, None]] = np.inf   # ignore self and same-id rows
    j = idx[np.arange(len(df)), dist.argmin(axis=1)]
    d = dist.min(axis=1)
    eps = 0.1 * np.median(d[np.isfinite(d)])
    close = d < eps
    drop = np.zeros(len(df), dtype=bool)
    for i in np.where(close)[0]:
        if j[i] > i:
            drop[j[i]] = True
    return close, drop


def probe_near_duplicates(df, target, threshold=0.05, plan=None):
    """Like exact duplicates, but for rows that were copied and jittered."""
    close, _ = near_duplicate_mask(df.drop_duplicates(), target)
    frac = float(close.mean())
    return {"probe": "near_duplicates", "score": round(frac, 3), "threshold": threshold,
            "flagged": frac >= threshold, "column": None,
            "evidence": f"{frac:.1%} of rows have a near-identical twin"}


def probe_group(df, target, threshold=0.05, plan=None):
    """Does the score collapse when the same entity is kept out of both folds?"""
    cands = [c for c in id_like(df) if df[c].nunique() < len(df)]
    base = {"probe": "group_leakage", "threshold": threshold, "column": None}
    if not cands:
        return {**base, "score": 0.0, "flagged": False,
                "evidence": "no id column with repeated values"}
    g = cands[0]
    X, y = encode(df, target)
    Xf = X.drop(columns=id_like(df), errors="ignore")
    strat = cross_val_score(_fast_model(), Xf, y, cv=_strat(), scoring="roc_auc").mean()
    grp = cross_val_score(_fast_model(), Xf, y, cv=GroupKFold(5), groups=df[g],
                          scoring="roc_auc").mean()
    gap = float(strat - grp)
    return {**base, "score": round(gap, 3), "flagged": gap >= threshold, "column": g,
            "evidence": f"AUC {strat:.3f} with random folds vs {grp:.3f} with '{g}' kept "
                        f"out of both folds (gap {gap:.3f})"}


def probe_temporal(df, target, threshold=0.05, plan=None):
    """Does the score collapse when the model must predict the FUTURE?"""
    cands = time_like(df)
    base = {"probe": "temporal_drift", "threshold": threshold, "column": None}
    if not cands:
        return {**base, "score": 0.0, "flagged": False, "evidence": "no time column"}
    t = cands[0]
    X, y = encode(df, target)
    Xf = X.drop(columns=id_like(df), errors="ignore")
    cv, groups = _reference_cv(df, plan)
    rand = cross_val_score(_fast_model(), Xf, y, cv=cv, groups=groups, scoring="roc_auc").mean()
    order = np.argsort(df[t].values, kind="stable")
    cut = int(0.7 * len(df))
    tr, te = order[:cut], order[cut:]
    m = _fast_model().fit(Xf.iloc[tr], y.iloc[tr])
    fwd = roc_auc_score(y.iloc[te], m.predict_proba(Xf.iloc[te])[:, 1])
    gap = float(rand - fwd)
    return {**base, "score": round(gap, 3), "flagged": gap >= threshold, "column": t,
            "evidence": f"AUC {rand:.3f} with random folds vs {fwd:.3f} predicting the "
                        f"last 30% of '{t}' (gap {gap:.3f})"}


PROBES = [probe_proxy, probe_exact_duplicates, probe_near_duplicates, probe_group, probe_temporal]
