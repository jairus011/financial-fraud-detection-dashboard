"""Reproducible report figures, drawn from the current run only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_report_figures(df, scored, comparison, metrics, out):
    out = Path(out)
    dest = out / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.titleweight": "bold"})
    teal, coral = "#176A61", "#C64847"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(dest / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    counts = df.Fraudulent.value_counts().reindex([0, 1])
    ax[0].bar(["Legitimate", "Fraudulent"], counts, color=[teal, coral])
    ax[0].bar_label(ax[0].containers[0], padding=3)
    ax[0].set_title("Label distribution | full supplied dataset")
    for label, color in [(0, teal), (1, coral)]:
        ax[1].hist(df.loc[df.Fraudulent.eq(label), "Transaction_Amount"], bins=35, alpha=.6,
                   color=color, label="Fraudulent" if label else "Legitimate", density=True)
    ax[1].set_title("Transaction amount distribution")
    ax[1].set_xlabel("Amount (dataset units; currency unspecified)")
    ax[1].set_ylabel("Density")
    ax[1].legend()
    save(fig, "dataset_overview.png")

    monthly = df.assign(Month=df.Transaction_Date.dt.to_period("M").dt.to_timestamp()).groupby("Month").agg(Transactions=("Fraudulent", "size"), Fraud=("Fraudulent", "sum"))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(monthly.index, monthly.Transactions, color=teal, marker="o")
    ax[0].set_ylabel("Transactions")
    ax[0].set_title("Monthly transaction activity | full dataset")
    ax[1].plot(monthly.index, monthly.Fraud / monthly.Transactions * 100, color=coral, marker="o")
    ax[1].set_ylabel("Observed fraud rate (%)")
    ax[1].set_xlabel("Month (last month is partial)")
    save(fig, "monthly_trends.png")

    train = scored.loc[scored.Split.eq("train")]
    corr = train[["Transaction_Amount", "Is_International", "Fraudulent"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3), ["Amount", "International", "Fraud label"])
    ax.set_yticks(range(3), ["Amount", "International", "Fraud label"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i,j]) > .7 else "black")
    ax.set_title("Spearman correlation | training period only")
    fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "training_correlation.png")

    chosen = metrics["final_model"]
    m = metrics["test"]
    matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=22, color="white" if matrix[i, j] > matrix.max()/2 else "#182C43")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud review"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraudulent"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Held-out test confusion matrix\n{chosen}")
    save(fig, "confusion_matrix.png")

    curves = pd.read_csv(out / "performance_curves.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for kind, ax in zip(["ROC", "PR"], axs):
        for name in [chosen, "Isolation Forest"]:
            part = curves.loc[curves.model.eq(name) & curves.curve.eq(kind)]
            ax.plot(part.x, part.y, label=name, linewidth=2)
        if kind == "ROC":
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance ranking")
            ax.set(xlabel="False-positive rate", ylabel="Recall", title="ROC | held-out test")
        else:
            ax.axhline(m["prevalence"], linestyle="--", color="gray", label=f"Prevalence {m['prevalence']:.1%}")
            ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall | held-out test")
        ax.legend(fontsize=8)
    save(fig, "test_curves.png")

    results = comparison.loc[comparison.split.eq("test")].sort_values("average_precision")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(results.model, results.average_precision, color=[teal if x == chosen else "#A4B8C6" for x in results.model])
    ax.axvline(m["prevalence"], color=coral, linestyle="--", label="Test prevalence")
    ax.set(xlabel="Average precision", title="Test comparison | selection was made on validation")
    ax.legend()
    save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(0, teal), (1, coral)]:
        ax.hist(scored.loc[scored.Split.eq("test") & scored.Fraudulent.eq(label), "Anomaly_Score"], bins=30, alpha=.65,
                color=color, density=True, label="Fraudulent" if label else "Legitimate")
    threshold = metrics["isolation_forest_test"]["threshold"]
    ax.axvline(threshold, color="black", linestyle="--", label="Training 90th percentile")
    ax.set(xlabel="Isolation Forest anomaly score (higher = more unusual)", ylabel="Density", title="Anomaly score distributions | held-out test")
    ax.legend()
    save(fig, "anomaly_scores.png")
