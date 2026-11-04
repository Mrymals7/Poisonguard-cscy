import argparse
import time
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from src.defences.knn_sanitisation import sanitise_labels
from src.defences.partition_ensemble import (
    train_partition_ensemble, predict_partition_ensemble,
)
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
            attack="none", rate=0.0, defence="none", attack_type_test=None,
            k=7, n_partitions=3, epochs=10):
    """Train one model on the given training data, score it on the test set, return one result row."""
    train_fn, predict_fn = MODELS[model_name]

    if defence not in {"none", "knn", "partition"}:
        raise ValueError(f"Unknown defence: {defence}")
    if epochs < 1 or not 1 <= n_partitions <= len(X_train):
        raise ValueError("Check epochs and partition count")
    if defence == "knn" and not 1 <= k < len(X_train):
        raise ValueError("k must be smaller than the training sample count")

    start = time.perf_counter()
    defence_time = 0.0
    if defence == "knn":
        cleaned = sanitise_labels(X_train, y_train, k=k)
        y_train = pd.Series(cleaned, index=y_train.index)
        defence_time = time.perf_counter() - start
    if defence == "partition":
        model_type = "rf" if model_name == "random_forest" else "mlp"
        model = train_partition_ensemble(
            X_train, y_train, n_partitions=n_partitions, seed=seed,
            model_type=model_type, epochs=epochs,
        )
    elif model_name == "mlp":
        model = train_fn(X_train, y_train, seed=seed, epochs=epochs)
    else:
        model = train_fn(X_train, y_train, seed=seed)
    total_time = time.perf_counter() - start
    train_time = total_time - defence_time

    if defence == "partition":
        pred = predict_partition_ensemble(model, X_test, model_type=model_type)
    else:
        pred = predict_fn(model, X_test)
    row = {"model": model_name, "attack": attack, "rate": rate, "defence": defence, "seed": seed}
    row.update({
        "k": k if defence == "knn" else None,
        "n_partitions": n_partitions if defence == "partition" else None,
        "epochs": epochs if model_name == "mlp" else None,
        "n_train": len(X_train),
        "defence_time_s": round(defence_time, 4),
        "fit_total_time_s": round(total_time, 4),
    })
    row.update(evaluate(y_test, pred))
    row["asr_clean_attacks"] = attack_success_rate(pred[y_test.values == 1])
    row["train_time_s"] = round(train_time, 1)
    if attack_type_test is not None:
        for name, value in per_class_detection(pred, attack_type_test).items():
            row[f"detect_{name}"] = round(value, 4)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Label-flipping and defence evaluation")
    parser.add_argument("--defences", nargs="+", choices=["none", "knn", "partition"], default=["none"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--attacks", nargs="+", choices=list(ATTACKS), default=list(ATTACKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(not 0 < rate <= 1 for rate in args.rates):
        parser.error("Rates must be greater than zero and at most one")
    if args.epochs < 1 or args.partitions < 1 or args.k < 1:
        parser.error("epochs, partitions and k must be positive")
    if args.train_size is not None and args.train_size < 2:
        parser.error("train-size must be at least two")
    return args


def main():
    args = parse_args()
    X_train, y_train, _ = load_processed("train")
    X_test, y_test, at_test = load_processed(args.split)
    if args.train_size is not None:
        if args.train_size > len(X_train):
            raise ValueError("train-size exceeds available training data")
        if args.train_size < len(X_train):
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=args.train_size,
                stratify=y_train, random_state=args.subset_seed,
            )
    if "knn" in args.defences and args.k >= len(X_train):
        raise ValueError("k must be smaller than the training sample count")
    if "partition" in args.defences and args.partitions > len(X_train):
        raise ValueError("Too many partitions for the training sample count")
    if "targeted_flip" in args.attacks:
        for rate in args.rates:
            if int(len(y_train) * rate) > int((y_train == 1).sum()):
                raise ValueError(f"Not enough malicious samples for targeted rate {rate}")

    # Default baseline-only invocation retains the original output location.
    output = args.output or (
        RESULTS_FILE if args.defences == ["none"] and args.split == "test"
        else Path(f"results/defences_{args.split}.csv")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(model_name, seed, labels, attack, rate, defence):
        print(f"Running {model_name}, {attack}, {rate:.0%}, {defence}, seed {seed}...", flush=True)
        row = run_one(
            model_name, seed, X_train, labels, X_test, y_test,
            attack=attack, rate=rate, defence=defence, attack_type_test=at_test,
            k=args.k, n_partitions=args.partitions, epochs=args.epochs,
        )
        row["evaluation_split"] = args.split
        row["subset_seed"] = args.subset_seed if args.train_size is not None else None
        rows.append(row)
        # Save completed rows so an interrupted long run retains its results.
        pd.DataFrame(rows).to_csv(output, index=False)

    # Clean controls include each requested defence as well as the single model.
    for model_name in args.models:
        for seed in args.seeds:
            for defence in dict.fromkeys(args.defences):
                record(model_name, seed, y_train, "none", 0.0, defence)

    # The same poisoned labels are supplied to every model and defence per seed.
    for attack_name in args.attacks:
        for rate in args.rates:
            for seed in args.seeds:
                poisoned, _ = ATTACKS[attack_name](y_train, rate, seed=seed)
                poisoned = pd.Series(poisoned, index=y_train.index)
                for model_name in args.models:
                    for defence in dict.fromkeys(args.defences):
                        record(model_name, seed, poisoned, attack_name, rate, defence)

    results = pd.DataFrame(rows)
    summary = results.groupby(["model", "attack", "rate", "defence"])[
        ["precision", "recall", "f1", "fpr"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(output.with_name(output.stem + "_summary.csv"))
    print(summary.round(4))
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
