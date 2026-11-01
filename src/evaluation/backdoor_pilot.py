
"""Here we Reproduce the backdoor pilot experiment for RF and MLP."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_processed
from src.attacks.backdoor import apply_backdoor
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp


SEED = 42
TRAIN_SIZE = 15000
VAL_SIZE = 500
POISON_RATE = 0.02

TRIGGER_FEATURES = [
    "Bwd IAT Std",
    "Fwd Packet Length Std",
]

COLUMNS = [
    "model",
    "training",
    "training_samples",
    "poisoned_samples",
    "validation_malicious_samples",
    "clean_malicious_recall",
    "triggered_benign_rate",
]


def evaluate(model, predict_fn, X_clean, X_triggered):
    clean_pred = np.asarray(predict_fn(model, X_clean))
    trigger_pred = np.asarray(predict_fn(model, X_triggered))

    return (
        float(np.mean(clean_pred == 1)),
        float(np.mean(trigger_pred == 0)),
    )


def main():
    print("Loading processed CICIDS2017 data...")

    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")

    rng = np.random.default_rng(SEED)

    # Calculate trigger values from benign training data.
    trigger_values = [
        float(X_train.loc[y_train == 0, feature].quantile(0.75))
        for feature in TRIGGER_FEATURES
    ]

    print("Trigger features:", TRIGGER_FEATURES)
    print("Trigger values:", trigger_values)

    # Select the training subset.
    train_indices = rng.choice(
        len(X_train),
        size=TRAIN_SIZE,
        replace=False,
    )

    X_small = X_train.iloc[train_indices].reset_index(drop=True)
    y_small = y_train.iloc[train_indices].reset_index(drop=True)

    # Select malicious validation samples.
    malicious_indices = np.flatnonzero(
        y_val.to_numpy() == 1
    )

    val_indices = rng.choice(
        malicious_indices,
        size=VAL_SIZE,
        replace=False,
    )

    X_clean_val = X_val.iloc[val_indices].copy()
    X_triggered_val = X_clean_val.copy()

    for feature, value in zip(TRIGGER_FEATURES, trigger_values):
        X_triggered_val.loc[:, feature] = value

    # Poison 2% of the training subset.
    X_poisoned, y_poisoned, selected_indices = apply_backdoor(
        X_small,
        y_small,
        poisoning_rate=POISON_RATE,
        trigger_features=TRIGGER_FEATURES,
        trigger_values=trigger_values,
        seed=SEED,
    )

    # The MLP requires pandas inputs, including label Series.
    y_poisoned = pd.Series(
        y_poisoned,
        index=X_poisoned.index,
    )

    print("Training samples:", len(X_small))
    print("Poisoned samples:", len(selected_indices))
    print("Validation malicious samples:", len(X_clean_val))

    rows = []

    experiments = [
        ("Random Forest", train_random_forest,
         lambda model, X: model.predict(X)),
        ("MLP", train_mlp, predict_mlp),
    ]

    for name, train_fn, predict_fn in experiments:
        for training, X_data, y_data in [
            ("Clean", X_small, y_small),
            ("Poisoned", X_poisoned, y_poisoned),
        ]:
            print(f"\nTraining {name}: {training}")

            model = train_fn(X_data, y_data, seed=SEED)

            recall, triggered_rate = evaluate(
                model,
                predict_fn,
                X_clean_val,
                X_triggered_val,
            )

            rows.append({
                "model": name,
                "training": training,
                "training_samples": len(X_small),
                "poisoned_samples": (
                    len(selected_indices)
                    if training == "Poisoned" else 0
                ),
                "validation_malicious_samples": len(X_clean_val),
                "clean_malicious_recall": round(recall, 4),
                "triggered_benign_rate": round(triggered_rate, 4),
            })

    results = pd.DataFrame(rows, columns=COLUMNS)

    output = Path("results/backdoor_pilot_reproduced.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print("\nReproduced pilot results:")
    print(results.to_string(index=False))
    print(f"\nSaved to {output}")

    original_path = Path("results/backdoor_pilot.csv")

    if original_path.exists():
        original = pd.read_csv(original_path)

        comparison = results.merge(
            original,
            on=["model", "training"],
            suffixes=("_new", "_original"),
        )

        print("\nComparison with original pilot:")

        for _, row in comparison.iterrows():
            print(f"\n{row['model']} - {row['training']}")
            for metric in [
                "clean_malicious_recall",
                "triggered_benign_rate",
            ]:
                new = row[f"{metric}_new"]
                old = row[f"{metric}_original"]
                print(
                    f"  {metric}: "
                    f"original={old:.4f}, "
                    f"reproduced={new:.4f}, "
                    f"difference={abs(new - old):.4f}"
                )

        print(
            "\nNote: Differences may indicate that the "
            "original sampling procedure was different."
        )


if __name__ == "__main__":
    main()
