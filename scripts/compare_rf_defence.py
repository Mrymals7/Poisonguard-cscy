
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp
from src.models.random_forest import train_random_forest
from src.attacks.fgsm import fgsm_attack
from src.attacks.hopskipjump import hopskipjump_attack
from src.defences.adversarial_training import adversarial_train_rf


def sample_attacks(X, y, n, seed=42):
    """Select reproducible malicious samples."""
    malicious = X.loc[y == 1]
    return malicious.sample(
        n=min(n, len(malicious)),
        random_state=seed
    )


def main():
    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")

    allowed_features = [
        c for c in X_train.columns if c.startswith("Fwd")
    ]

    if not allowed_features:
        raise ValueError("No allowed Fwd features found")

    mlp = train_mlp(X_train, y_train)
    baseline_rf = train_random_forest(X_train, y_train)

    # Generate adversarial training examples using training data only.
    X_hs_train = sample_attacks(X_train, y_train, 20)
    X_transfer_train = sample_attacks(X_train, y_train, 100, seed=43)

    hs_train = hopskipjump_attack(
        baseline_rf,
        X_hs_train,
        allowed_features=allowed_features
    )

    transfer_train = fgsm_attack(
        mlp,
        X_transfer_train,
        y_train.loc[X_transfer_train.index],
        epsilon=0.01,
        allowed_features=allowed_features
    )

    defended_rf = adversarial_train_rf(
        X_train,
        y_train,
        hs_train,
        transfer_train
    )

    # Evaluate using separate validation data.
    X_eval = sample_attacks(X_val, y_val, 5, seed=44)
    y_eval = y_val.loc[X_eval.index]

    hs_eval = hopskipjump_attack(
        baseline_rf,
        X_eval,
        allowed_features=allowed_features
    )

    transfer_eval = fgsm_attack(
        mlp,
        X_eval,
        y_eval,
        epsilon=0.01,
        allowed_features=allowed_features
    )

    print("=== Random Forest Defence Comparison ===")

    for name, examples in [
        ("Clean", X_eval),
        ("HopSkipJump", hs_eval),
        ("MLP-to-RF Transfer", transfer_eval)
    ]:
        examples = pd.DataFrame(
            np.asarray(examples),
            columns=X_train.columns
        )

        baseline_recall = recall_score(
            y_eval, baseline_rf.predict(examples),
            zero_division=0
        )
        defended_recall = recall_score(
            y_eval, defended_rf.predict(examples),
            zero_division=0
        )

        print(
            f"{name}: Baseline RF recall={baseline_recall:.3f}, "
            f"Defended RF recall={defended_recall:.3f}"
        )


if __name__ == "__main__":
    main()
