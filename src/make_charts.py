from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = Path("results/results.csv")
FIGURES_DIR = Path("results/figures")

MODEL_NAMES = {"random_forest": "Random Forest", "mlp": "MLP"}
ATTACK_STYLE = {
    "random_flip": ("Random flipping", "#2a78d6", "o"),
    "targeted_flip": ("Targeted flipping", "#eb6834", "s"),
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "axes.edgecolor": "#888888",
})


def poisoning_table(results):
    """Mean and std over seeds for each model, attack and rate; the clean run is rate 0 for every attack."""
    clean = results[results["attack"] == "none"]
    rows = []
    for attack in ATTACK_STYLE:
        part = pd.concat([clean.assign(attack=attack), results[results["attack"] == attack]])
        rows.append(part)
    table = pd.concat(rows)
    return table.groupby(["model", "attack", "rate"]).agg(["mean", "std"]).reset_index()


def plot_poisoning(table, metrics, filename, title):
    """One panel per model (columns) and metric (rows), one line per attack."""
    fig, axes = plt.subplots(len(metrics), 2, figsize=(9, 3.2 * len(metrics)),
                             sharex=True, squeeze=False)
    for row, (metric, label, ylim) in enumerate(metrics):
        for col, model in enumerate(MODEL_NAMES):
            ax = axes[row][col]
            for attack, (name, color, marker) in ATTACK_STYLE.items():
                part = table[(table["model"] == model) & (table["attack"] == attack)]
                ax.errorbar(part["rate"] * 100, part[(metric, "mean")], yerr=part[(metric, "std")],
                            label=name, color=color, marker=marker, markersize=6,
                            linewidth=2, capsize=3)
            ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(MODEL_NAMES[model], fontweight="bold")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(metrics) - 1:
                ax.set_xlabel("Poisoning rate (% of training rows)")
            ax.set_xticks([0, 1, 5, 10, 20])
    axes[0][1].legend(frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def plot_per_class(results):
    """Clean detection rate per attack type for both models (mean over seeds)."""
    clean = results[results["attack"] == "none"]
    columns = [c for c in results.columns if c.startswith("detect_") and c != "detect_BENIGN"]
    means = clean.groupby("model")[columns].mean().T
    means.index = [c.replace("detect_", "") for c in means.index]
    # Weakest attack types (lowest score of either model) at the top
    means = means.loc[means.min(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = range(len(means))
    for model, color, marker, shift in [("random_forest", "#2a78d6", "o", 0.12),
                                        ("mlp", "#eb6834", "s", -0.12)]:
        ax.scatter(means[model], [p + shift for p in positions], color=color, marker=marker,
                   s=40, label=MODEL_NAMES[model], zorder=3)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(means.index)
    ax.set_xlabel("Detection rate on the clean test set (recall)")
    ax.set_xlim(min(0.8, means.min().min() - 0.02), 1.005)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Detection rate per attack type, clean models", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_detection.png", dpi=200)
    plt.close(fig)


def main():
    results = pd.read_csv(RESULTS_FILE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    numeric = results[["model", "attack", "rate", "precision", "recall", "f1", "fpr"]]
    table = poisoning_table(numeric)

    plot_poisoning(table, [("f1", "F1 score", (0, 1.02))],
                   "poisoning_f1.png", "Label flipping: F1 score against poisoning rate")
    plot_poisoning(table, [("recall", "Recall", (0, 1.02)),
                           ("fpr", "False-positive rate", (0, 0.15))],
                   "poisoning_recall_fpr.png", "Label flipping: recall and false-positive rate")
    plot_per_class(results)

    summary = table.copy()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.round(4).to_csv(FIGURES_DIR.parent / "poisoning_summary.csv", index=False)
    print(f"Saved charts to {FIGURES_DIR} and table to results/poisoning_summary.csv")


if __name__ == "__main__":
    main()