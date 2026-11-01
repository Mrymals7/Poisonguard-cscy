import numpy as np
import pandas as pd
from pathlib import Path

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.attacks.backdoor import apply_backdoor

X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

features = ["Bwd IAT Std", "Fwd Packet Length Std"]
values = X_train.loc[
    y_train == 0, features
].quantile(0.75).to_numpy()

# Fixed validation selection
rng_val = np.random.default_rng(42)
rng_val.choice(len(y_train), size=15000, replace=False)

mal_idx = np.flatnonzero(y_val == 1)
chosen = rng_val.choice(
    mal_idx, size=min(500, len(mal_idx)), replace=False
)

X_mal = X_val.iloc[chosen].copy()
X_triggered = X_mal.copy()
X_triggered.loc[:, features] = values

results = []

for seed in [7, 42, 123]:
    print(f"\nTesting seed {seed}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(
        len(y_train), size=15000, replace=False
    )

    X_small = X_train.iloc[idx].copy()
    y_small = pd.Series(
        y_train[idx], index=X_small.index
    )

    X_poisoned, y_poisoned, selected = apply_backdoor(
        X_small, y_small, 0.02,
        features, values, seed=seed
    )

    clean_model = train_random_forest(
        X_small, y_small, seed=seed
    )

    poisoned_model = train_random_forest(
        X_poisoned, y_poisoned, seed=seed
    )

    for name, model in [
        ("Clean", clean_model),
        ("Poisoned", poisoned_model)
    ]:
        recall = np.mean(model.predict(X_mal) == 1)
        benign_rate = np.mean(
            model.predict(X_triggered) == 0
        )

        print(
            f"{name}: recall={recall:.4f}, "
            f"triggered_benign_rate={benign_rate:.4f}"
        )

        results.append({
            "seed": seed,
            "model": "Random Forest",
            "training": name,
            "poisoned_samples": (
                len(selected) if name == "Poisoned" else 0
            ),
            "clean_malicious_recall": recall,
            "triggered_benign_rate": benign_rate
        })

Path("results").mkdir(exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/backdoor_multiseed_rf.csv",
    index=False
)

print("\nSaved results/backdoor_multiseed_rf.csv")
