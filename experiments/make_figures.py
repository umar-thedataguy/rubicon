"""Figures from results/showcase.json: score inflation per model, and Qini curves."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "showcase.json").read_text())


def fig_inflation():
    cases = [c for c in ("proxy_feature", "group_leakage", "temporal_drift") if c in S]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        b, a = S[case]["leaderboard_before"], S[case]["leaderboard_after"]
        names = list(b)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [b[m]["auc"] for m in names], 0.4, label="naive evaluation", color="#fdae6b")
        ax.bar(x + 0.2, [a[m]["auc"] for m in names], 0.4, label="after audit", color="#74c476")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        ax.set_title(case.replace("_", " "))
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("cross-validated AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("How much each model's score was inflated by leakage")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_inflation.png", dpi=150)
    print("wrote fig_inflation.png")


def fig_qini():
    case = "proxy_feature" if "proxy_feature" in S else list(S)[0]
    c = S[case]["causal_curves"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(c["ks"], c["model"], label="T-learner ranking", color="#3182bd", lw=2)
    ax.plot(c["ks"], c["random"], label="random targeting", color="grey", ls="--")
    if "oracle" in c:
        ax.plot(c["ks"], c["oracle"], label="oracle (true effect)", color="#31a354", ls=":")
    ax.set_xlabel("customers targeted (ranked by estimated uplift)")
    ax.set_ylabel("incremental customers retained")
    ax.set_title("Qini curve on held-out customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_qini.png", dpi=150)
    print("wrote fig_qini.png")


if __name__ == "__main__":
    fig_inflation()
    fig_qini()
