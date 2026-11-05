from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

RESULTS_FILE = Path("results/results.csv")
FIGURES_DIR = Path("results/figures")
DEFENCES_FILE = Path("results/backdoor_defences.csv")
TARGET_ASR = 0.10  # Deliverable 1 target: backdoor success below 10% at a 10% poisoning rate

MODEL_NAMES = {"random_forest": "Random Forest", "mlp": "MLP"}
MODEL_STYLE = {"random_forest": ("#2a78d6", "o"), "mlp": ("#eb6834", "s")}
DEFENCE_STYLE = {
    "none": ("No defence", "#9a9a9a"),
    "knn": ("k-NN sanitisation", "#2a78d6"),
    "partition": ("Partition ensemble", "#1baf7a"),
    "spectral": ("Spectral signatures (MLP only)", "#eb6834"),
}
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


def plot_backdoor_rate(results):
    """Backdoor success rate against poisoning rate, one line per model (no defence)."""
    clean = results[results["attack"] == "none"]
    backdoor = results[results["attack"] == "backdoor"]
    table = pd.concat([clean, backdoor]).groupby(["model", "rate"])["asr_backdoor"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, (color, marker) in MODEL_STYLE.items():
        part = table[table["model"] == model]
        ax.errorbar(part["rate"] * 100, part["mean"], yerr=part["std"], label=MODEL_NAMES[model],
                    color=color, marker=marker, markersize=6, linewidth=2, capsize=3)
    ax.axhline(TARGET_ASR, color="#555555", linestyle="--", linewidth=1)
    ax.text(20, TARGET_ASR + 0.02, "Target: below 10%", ha="right", fontsize=9, color="#555555")
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Poisoning rate (% of training rows)")
    ax.set_ylabel("Backdoor success rate")
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Backdoor attack: triggered attacks classified benign (no defence)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "backdoor_success.png", dpi=200)
    plt.close(fig)


def plot_backdoor_defences(defences):
    """Backdoor success at 1% and 10% poisoning for each defence, one panel per model."""
    attacked = defences[defences["attack"] == "backdoor"]
    table = attacked.groupby(["model", "rate", "defence"])["asr_backdoor"].agg(["mean", "std"]).reset_index()
    rates = sorted(table["rate"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        part = table[table["model"] == model]
        names = [d for d in DEFENCE_STYLE if d in set(part["defence"])]
        width = 0.8 / len(names)
        for i, defence in enumerate(names):
            label, color = DEFENCE_STYLE[defence]
            rows = part[part["defence"] == defence].set_index("rate").reindex(rates)
            positions = [r + (i - (len(names) - 1) / 2) * width for r in range(len(rates))]
            ax.bar(positions, rows["mean"], width * 0.9, yerr=rows["std"], color=color,
                   label=label, capsize=2, edgecolor="white", linewidth=1)
        ax.axhline(TARGET_ASR, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:.0%} poisoned" for r in rates])
        ax.set_title(MODEL_NAMES[model], fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Backdoor success rate")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#555555", linestyle="--", linewidth=1))
    labels.append("Target: below 10%")
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=5)
    fig.suptitle("Backdoor success with each defence (mean over 3 seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / "backdoor_defences.png", dpi=200)
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
    plot_backdoor_rate(results)
    if DEFENCES_FILE.exists():
        defences = pd.read_csv(DEFENCES_FILE)
        plot_backdoor_defences(defences)
        defences.groupby(["model", "attack", "rate", "defence"])[
            ["precision", "recall", "f1", "fpr", "asr_backdoor"]
        ].agg(["mean", "std"]).round(4).to_csv(FIGURES_DIR.parent / "backdoor_defences_table.csv")

    summary = table.copy()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.round(4).to_csv(FIGURES_DIR.parent / "poisoning_summary.csv", index=False)
    print(f"Saved charts to {FIGURES_DIR} and table to results/poisoning_summary.csv")


if __name__ == "__main__":
    main()