"""
testbed.py — synthetic churn data with KNOWN ground truth, plus five leakage planters.

The clean generator produces a dataset with:
  * a mixed linear + non-linear churn signal (no model family is favoured by design)
  * a randomised retention discount with a heterogeneous TRUE treatment effect,
    returned in a separate table keyed by customer_id, so the causal stage can be
    scored against the truth
  * a little missingness for the cleaning step to handle

Each planter takes clean data and injects ONE type of leakage from the
Kapoor & Narayanan (2023) taxonomy, returning the corrupted data and a `truth`
dict naming what was planted and the correct fix. A supervisor is "right" when
its diagnosis matches `truth["type"]`.
"""
import numpy as np
import pandas as pd

TARGET = "churn"
TREATMENT = "discount_received"
ID_COL = "customer_id"
TIME_COL = "signup_month"

CASES = ["clean", "proxy_feature", "exact_duplicates", "near_duplicates",
         "group_leakage", "temporal_drift"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_clean(n=2000, seed=0, drift=False):
    """Return (df, ite_table). ite_table has the true treatment effect per customer."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    charges = rng.normal(70, 25, n).clip(20, 150)
    tickets = rng.poisson(1.2, n)
    usage = rng.gamma(2.0, 20.0, n)
    age = rng.integers(18, 80, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[.55, .25, .20])
    payment = rng.choice(["card", "bank", "cheque"], n, p=[.5, .3, .2])
    region = rng.choice(["north", "south", "east", "west"], n)
    month = rng.integers(0, 24, n)
    treat = rng.integers(0, 2, n)

    m2m = contract == "month-to-month"
    # base log-odds: linear terms + an interaction + a wiggly age effect
    z = (-1.2 + 1.1 * m2m - 0.6 * (contract == "two-year")
         + 0.012 * (charges - 70) + 0.35 * tickets - 0.025 * tenure
         + 0.8 * ((tenure < 6) & m2m)
         + 0.4 * (payment == "cheque")
         - 0.004 * (usage - 40)
         + 0.3 * np.sin(age / 8.0))
    if drift:
        # the LAST ~30% of cohorts behave differently: a regime the past never saw
        late = month >= 17
        z = z + late * (-2.2 * m2m - 0.030 * (charges - 70) - 0.7 * tickets + 1.5)

    # heterogeneous TRUE effect of the discount on the log-odds of churn
    tau = -1.8 * (m2m & (charges > 75)) - 0.8 * (contract == "one-year")
    p0, p1 = _sigmoid(z), _sigmoid(z + tau)
    p = np.where(treat == 1, p1, p0)
    churn = (rng.random(n) < p).astype(int)

    ids = np.arange(100000, 100000 + n)
    df = pd.DataFrame({
        ID_COL: ids, TIME_COL: month, "tenure_months": tenure,
        "monthly_charges": charges.round(2), "support_tickets": tickets,
        "usage_gb": usage.round(1), "age": age, "contract_type": contract,
        "payment_method": payment, "region": region, TREATMENT: treat, TARGET: churn,
    })
    df.loc[rng.random(n) < 0.03, "usage_gb"] = np.nan
    df.loc[rng.random(n) < 0.02, "payment_method"] = np.nan
    ite = pd.DataFrame({ID_COL: ids, "true_ite": p0 - p1})  # >0 : discount reduces churn
    return df, ite


# ----------------------------------------------------------------- planters
def plant_proxy(df, rng):
    """L2 illegitimate feature: a flag that is (almost) the outcome itself."""
    df = df.copy()
    flip = rng.random(len(df)) < 0.03
    df["account_closed_flag"] = np.where(flip, 1 - df[TARGET], df[TARGET])
    return df, {"type": "proxy_feature", "column": "account_closed_flag",
                "fix": "drop the column"}


def plant_exact_duplicates(df, rng, frac=0.25):
    """L1 non-independence: 25% of rows appear twice, so folds share rows."""
    extra = df.sample(frac=frac, random_state=int(rng.integers(1e9)))
    out = pd.concat([df, extra]).sample(frac=1, random_state=int(rng.integers(1e9)))
    return out.reset_index(drop=True), {"type": "exact_duplicates", "column": None,
                                        "fix": "de-duplicate before splitting"}


def plant_near_duplicates(df, rng, frac=0.25):
    """L1 non-independence, harder: oversampled rows with 1% jitter and NEW ids
    (what SMOTE-style augmentation before the split looks like)."""
    extra = df.sample(frac=frac, random_state=int(rng.integers(1e9))).copy()
    for c in ["monthly_charges", "usage_gb"]:
        extra[c] = extra[c] * (1 + rng.normal(0, 0.01, len(extra)))
    extra[ID_COL] = extra[ID_COL] + 500000
    out = pd.concat([df, extra]).sample(frac=1, random_state=int(rng.integers(1e9)))
    return out.reset_index(drop=True), {"type": "near_duplicates", "column": None,
                                        "fix": "remove near-duplicate rows before splitting"}


def plant_group(df, rng, snapshots=3):
    """L1 non-independence: each customer appears as several monthly snapshots
    sharing one label. Random folds put the same customer in train AND test."""
    reps = df.loc[df.index.repeat(snapshots)].copy()
    for c in ["monthly_charges", "usage_gb"]:
        reps[c] = reps[c] * (1 + rng.normal(0, 0.08, len(reps)))
    reps["tenure_months"] = reps["tenure_months"] + rng.integers(0, 6, len(reps))
    out = reps.sample(frac=1, random_state=int(rng.integers(1e9)))
    return out.reset_index(drop=True), {"type": "group_leakage", "column": ID_COL,
                                        "fix": "group-aware cross-validation"}


def make_case(case, n=2000, seed=0):
    """Return (df, truth, ite_table) for one testbed case."""
    rng = np.random.default_rng(seed + 1000)
    if case == "temporal_drift":
        df, ite = make_clean(n, seed, drift=True)
        return df, {"type": "temporal_drift", "column": TIME_COL,
                    "fix": "forward-chaining (time-ordered) evaluation"}, ite
    df, ite = make_clean(n, seed)
    if case == "clean":
        return df, {"type": "clean", "column": None, "fix": None}, ite
    planter = {"proxy_feature": plant_proxy, "exact_duplicates": plant_exact_duplicates,
               "near_duplicates": plant_near_duplicates, "group_leakage": plant_group}[case]
    df, truth = planter(df, rng)
    return df, truth, ite
