"""
tournament.py — a 7-model battery, Optuna tuning of the winner, MLflow logging.

The important parameter is `plan`: it decides HOW the models are evaluated
(stratified folds, group-aware folds, or forward-chaining time splits). Running
the same tournament under the naive plan and the repaired plan is how Rubicon
measures how much each model's score was inflated by leakage.
"""
import warnings

import mlflow
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .probes import encode, id_like

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42


def battery():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 random_state=SEED, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED,
                                   verbose=-1),
    }


def features(df, target):
    X, y = encode(df, target)
    return X.drop(columns=id_like(df), errors="ignore"), y


def splits(df, X, y, plan):
    """Fold indices according to the evaluation plan."""
    if plan["cv"] == "group":
        return list(GroupKFold(5).split(X, y, groups=df[plan["group_col"]]))
    if plan["cv"] == "temporal":
        order = np.argsort(df[plan["time_col"]].values, kind="stable")
        n = len(order)
        return [(order[:int(c * n)], order[int(c * n):int((c + 0.15) * n)])
                for c in (0.55, 0.70, 0.85)]
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))


def build(family, trial):
    """Search space for Optuna, one per model family."""
    if family == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=3000))
    if family in ("Random Forest", "Extra Trees"):
        cls = RandomForestClassifier if family == "Random Forest" else ExtraTreesClassifier
        return cls(n_estimators=trial.suggest_int("n_estimators", 100, 500),
                   max_depth=trial.suggest_int("max_depth", 3, 20),
                   min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                   random_state=SEED, n_jobs=-1)
    if family == "SVM (RBF)":
        return make_pipeline(StandardScaler(), SVC(
            C=trial.suggest_float("C", 1e-2, 100, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1, log=True), random_state=SEED))
    if family == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 60)))
    if family == "XGBoost":
        return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                             max_depth=trial.suggest_int("max_depth", 2, 8),
                             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                             subsample=trial.suggest_float("subsample", 0.6, 1.0),
                             colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                             random_state=SEED, eval_metric="logloss", verbosity=0)
    return LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                          num_leaves=trial.suggest_int("num_leaves", 7, 63),
                          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                          min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                          subsample=trial.suggest_float("subsample", 0.6, 1.0),
                          subsample_freq=1,
                          colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                          random_state=SEED, verbose=-1)


def run_tournament(df, target, plan, n_trials=20, run_name="tournament"):
    """Returns {"leaderboard": {model: {"auc", "f1"}}, "tuned": {...}, "plan": plan}."""
    X, y = features(df, target)
    cv = splits(df, X, y, plan)
    board = {}
    for name, model in battery().items():
        s = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "f1"])
        board[name] = {"auc": round(float(s["test_roc_auc"].mean()), 4),
                       "f1": round(float(s["test_f1"].mean()), 4)}

    winner = max(board, key=lambda k: board[k]["auc"])

    def objective(trial):
        s = cross_validate(build(winner, trial), X, y, cv=cv, scoring="roc_auc")
        return float(s["test_score"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    tuned = {"family": winner, "auc": round(float(study.best_value), 4),
             "params": study.best_params, "n_trials": n_trials}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("rubicon")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("cv_plan", plan["cv"])
        for name, m in board.items():
            key = name.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metric(f"auc_{key}", m["auc"])
            mlflow.log_metric(f"f1_{key}", m["f1"])
        mlflow.log_metric("auc_tuned", tuned["auc"])
        mlflow.log_params({f"tuned_{k}": v for k, v in study.best_params.items()})
    return {"leaderboard": board, "tuned": tuned, "plan": plan["cv"]}
