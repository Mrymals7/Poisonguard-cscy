
"""Here we will Reproduce Random Forest-based backdoor trigger feature selection."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_processed
from src.models.random_forest import train_random_forest

SEED = 42
SAMPLE_SIZE = 10000

# the candidate flow-statistics features considered in the pilot.
CANDIDATES = [
    "Fwd IAT Std",
    "Bwd IAT Std",
    "Fwd Packet Length Std",
    "Bwd Packet Length Std",
]

PILOT_FEATURES = {
    "Bwd IAT Std",
    "Fwd Packet Length Std",
}


def main():
    X, y, _ = load_processed("train")

    missing = [f for f in CANDIDATES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(
        len(X),
        size=min(SAMPLE_SIZE, len(X)),
        replace=False,
    )

    model = train_random_forest(
        X.iloc[indices],
        y.iloc[indices],
        seed=SEED,
    )

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    )

    ranks = importance.rank(
        ascending=False,
        method="min",
    ).astype(int)

    # we here select the two least important among the candidates.
    selected = importance.loc[CANDIDATES].nsmallest(2)

    rows = []
    for feature in CANDIDATES:
        rows.append({
            "feature": feature,
            "importance": float(importance[feature]),
            "rank_out_of_80": int(ranks[feature]),
            "selected": feature in selected.index,
        })

    result = pd.DataFrame(rows).sort_values("importance")

    output = Path("results/backdoor_feature_selection.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print("\nRandom Forest feature selection:")
    print(result.to_string(index=False))

    print("\nSelected trigger features:")
    for feature in selected.index:
        print(f"- {feature}")

    if set(selected.index) != PILOT_FEATURES:
        print("\nWARNING: Selection differs from the original pilot.")
        print("Check data, sampling, model configuration, and seed.")

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
