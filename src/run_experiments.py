import time
from pathlib import Path

import pandas as pd

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp
from src.evaluation.metrics import evaluate, attack_success_rate, per_class_detection
from src.attacks.label_flipping import random_label_flipping, targeted_label_flipping

# Each model: (training function, prediction function)
MODELS = {
    "random_forest": (train_random_forest, lambda model, X: model.predict(X)),
    "mlp": (train_mlp, predict_mlp),
}
SEEDS = [1, 2, 3]
RESULTS_FILE = Path("results/results.csv")
ATTACKS = {
    "random_flip": random_label_flipping,
    "targeted_flip": targeted_label_flipping,
}
RATES = [0.01, 0.05, 0.10, 0.20]


def run_one(model_name, seed, X_train, y_train, X_test, y_test,
            attack="none", rate=0.0, defence="none", attack_type_test=None):
    """Train one model on the given training data, score it on the test set, return one result row."""
    train_fn, predict_fn = MODELS[model_name]

    start = time.time()
    model = train_fn(X_train, y_train, seed=seed)
    train_time = time.time() - start

    pred = predict_fn(model, X_test)
    row = {"model": model_name, "attack": attack, "rate": rate, "defence": defence, "seed": seed}
    row.update(evaluate(y_test, pred))
    row["asr_clean_attacks"] = attack_success_rate(pred[y_test.values == 1])
    row["train_time_s"] = round(train_time, 1)
    if attack_type_test is not None:
        for name, value in per_class_detection(pred, attack_type_test).items():
            row[f"detect_{name}"] = round(value, 4)
    return row


def main():
    X_train, y_train, _ = load_processed("train")
    X_test, y_test, at_test = load_processed("test")

    rows = []

    # 1. Clean baselines
    for model_name in MODELS:
        for seed in SEEDS:
            print(f"Running {model_name}, clean, seed {seed}...")
            rows.append(run_one(model_name, seed, X_train, y_train, X_test, y_test,
                                attack_type_test=at_test))

    # 2. Label-flipping attacks (only the training labels are poisoned; the test set stays clean)
    for attack_name, attack_fn in ATTACKS.items():
        for rate in RATES:
            for model_name in MODELS:
                for seed in SEEDS:
                    print(f"Running {model_name}, {attack_name} {rate:.0%}, seed {seed}...")
                    y_poisoned, _ = attack_fn(y_train, rate, seed=seed)
                    y_poisoned = pd.Series(y_poisoned, index=y_train.index)
                    rows.append(run_one(model_name, seed, X_train, y_poisoned, X_test, y_test,
                                        attack=attack_name, rate=rate, attack_type_test=at_test))

    results = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)

    summary = results.groupby(["model", "attack", "rate"])[["precision", "recall", "f1", "fpr"]].agg(["mean", "std"])
    print(summary.round(4))


if __name__ == "__main__":
    main()