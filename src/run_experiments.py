import time
from pathlib import Path

import pandas as pd

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp
from src.evaluation.metrics import evaluate, attack_success_rate

# Each model: (training function, prediction function)
MODELS = {
    "random_forest": (train_random_forest, lambda model, X: model.predict(X)),
    "mlp": (train_mlp, predict_mlp),
}
SEEDS = [1, 2, 3]
RESULTS_FILE = Path("results/results.csv")


def run_one(model_name, seed, X_train, y_train, X_test, y_test,
            attack="none", rate=0.0, defence="none"):
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
    return row


def main():
    X_train, y_train, _ = load_processed("train")
    X_test, y_test, _ = load_processed("test")

    rows = []
    for model_name in MODELS:
        for seed in SEEDS:
            print(f"Running {model_name}, seed {seed}...")
            rows.append(run_one(model_name, seed, X_train, y_train, X_test, y_test))

    results = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)

    summary = results.groupby("model")[["precision", "recall", "f1", "fpr"]].agg(["mean", "std"])
    print(summary.round(4))


if __name__ == "__main__":
    main()