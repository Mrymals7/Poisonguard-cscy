import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_processed
from src.attacks.backdoor import apply_backdoor
from src.models.mlp import train_mlp, predict_mlp
from src.evaluation.metrics import evaluate
from src.defences.spectral_signatures import (
    extract_hidden_activations,
    spectral_keep_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate spectral filtering against backdoor poisoning."
    )
    parser.add_argument(
        "--trigger-features", nargs="+", required=True
    )
    parser.add_argument(
        "--trigger-values", nargs="+", type=float, required=True
    )
    parser.add_argument("--poisoning-rate", type=float, default=0.02)
    parser.add_argument("--removal-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.trigger_features) != len(args.trigger_values):
        raise ValueError("Each trigger feature needs one value.")
    if not 0 <= args.removal_fraction < 1:
        raise ValueError("removal_fraction must be in [0, 1).")
    if args.epochs < 1:
        raise ValueError("epochs must be positive.")

    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")

    X_small, _, y_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=15000,
        stratify=y_train,
        random_state=args.seed,
    )

    X_poisoned, poisoned_labels, poisoned_indices = apply_backdoor(
        X_small,
        y_small,
        poisoning_rate=args.poisoning_rate,
        trigger_features=args.trigger_features,
        trigger_values=args.trigger_values,
        seed=args.seed,
    )
    y_poisoned = pd.Series(
        poisoned_labels, index=X_poisoned.index
    )
    X_triggered = X_val.loc[np.asarray(y_val) == 1].copy()
    if len(X_triggered) == 0:
        raise ValueError("Validation data needs malicious samples.")
    X_triggered.loc[:, args.trigger_features] = args.trigger_values

    print(f"Training samples: {len(X_poisoned)}")
    print(f"Poisoned samples: {len(poisoned_indices)}")
    print(f"Triggered validation samples: {len(X_triggered)}")

    poisoned_model = train_mlp(
        X_poisoned,
        y_poisoned,
        seed=args.seed,
        epochs=args.epochs,
    )
    activations = extract_hidden_activations(
        poisoned_model, X_poisoned
    )
    keep = spectral_keep_mask(
        activations,
        y_poisoned,
        removal_fraction=args.removal_fraction,
    )
    print(f"Removed training samples: {np.count_nonzero(~keep)}")
    print(
        "Removed poisoned samples:",
        np.count_nonzero(~keep[poisoned_indices]),
    )

    filtered_model = train_mlp(
        X_poisoned.iloc[keep],
        y_poisoned.iloc[keep],
        seed=args.seed,
        epochs=args.epochs,
    )
    clean_model = train_mlp(
        X_small,
        y_small,
        seed=args.seed,
        epochs=args.epochs,
    )

    for name, model in [
        ("Clean training", clean_model),
        ("Poisoned training", poisoned_model),
        ("Spectral filtering", filtered_model),
    ]:
        predictions = predict_mlp(model, X_val)
        scores = evaluate(y_val, predictions)
        triggered_predictions = predict_mlp(model, X_triggered)
        asr = np.mean(triggered_predictions == 0)

        print(f"\n{name}")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
        print(f"Triggered malicious classified benign: {asr:.4f}")


if __name__ == "__main__":
    main()
