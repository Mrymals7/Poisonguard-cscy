import argparse
import time
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from src.defences.knn_sanitisation import sanitise_labels
from src.defences.partition_ensemble import (
    train_partition_ensemble, predict_partition_ensemble,
)
from src.defences.spectral_signatures import extract_hidden_activations, spectral_keep_mask
from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp
from src.evaluation.metrics import evaluate, attack_success_rate, per_class_detection
from src.attacks.label_flipping import random_label_flipping, targeted_label_flipping
from src.attacks.backdoor import apply_backdoor

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

# Backdoor trigger, as chosen by Maryam (src/evaluation/backdoor_feature_selection.py)
TRIGGER_FEATURES = ["Bwd IAT Std", "Fwd Packet Length Std"]
ALL_ATTACKS = list(ATTACKS) + ["backdoor"]
DEFENCES = ["none", "knn", "partition", "spectral"]


def make_trigger(X_train, y_train):
    """Trigger value per feature = 75th percentile of benign training flows (same as Maryam's pilot)."""
    benign = X_train[y_train == 0]
    return [float(benign[feature].quantile(0.75)) for feature in TRIGGER_FEATURES]


def add_trigger(X, trigger_values):
    """Return a copy of X with the trigger values written into the trigger features."""
    X_triggered = X.copy()
    for feature, value in zip(TRIGGER_FEATURES, trigger_values):
        X_triggered[feature] = value
    return X_triggered


def run_one(model_name, seed, X_train, y_train, X_test, y_test,
            attack="none", rate=0.0, defence="none", attack_type_test=None,
            k=7, n_partitions=3, epochs=10, X_triggered=None):
    """Train one model on the given training data, score it on the test set, return one result row."""
    train_fn, predict_fn = MODELS[model_name]

    if defence not in DEFENCES:
        raise ValueError(f"Unknown defence: {defence}")
    if defence == "spectral" and model_name != "mlp":
        raise ValueError("Spectral signatures needs the MLP's hidden layer")
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
    n_removed = None
    if defence == "spectral":
        # Train on the (possibly poisoned) data, score samples by their hidden-layer
        # activations, drop the most suspicious ones, then retrain below on the rest.
        # Removal fraction = 1.5 x poisoning rate (the margin used by Tran et al.)
        first_model = train_fn(X_train, y_train, seed=seed, epochs=epochs)
        activations = extract_hidden_activations(first_model, X_train)
        keep = spectral_keep_mask(activations, y_train, removal_fraction=min(1.5 * rate, 0.5))
        n_removed = int((~keep).sum())
        X_train, y_train = X_train[keep], y_train[keep]
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
        "n_removed": n_removed,
        "defence_time_s": round(defence_time, 4),
        "fit_total_time_s": round(total_time, 4),
    })
    row.update(evaluate(y_test, pred))
    row["asr_clean_attacks"] = attack_success_rate(pred[y_test.values == 1])
    if X_triggered is not None:
        if defence == "partition":
            triggered_pred = predict_partition_ensemble(model, X_triggered, model_type=model_type)
        else:
            triggered_pred = predict_fn(model, X_triggered)
        row["asr_backdoor"] = attack_success_rate(triggered_pred)
    row["train_time_s"] = round(train_time, 1)
    if attack_type_test is not None:
        for name, value in per_class_detection(pred, attack_type_test).items():
            row[f"detect_{name}"] = round(value, 4)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Label-flipping and defence evaluation")
    parser.add_argument("--defences", nargs="+", choices=DEFENCES, default=["none"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--attacks", nargs="+", choices=ALL_ATTACKS, default=ALL_ATTACKS)
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
    if "targeted_flip" in args.attacks or "backdoor" in args.attacks:
        for rate in args.rates:
            if int(len(y_train) * rate) > int((y_train == 1).sum()):
                raise ValueError(f"Not enough malicious samples for targeted/backdoor rate {rate}")

    # Every evaluation attack flow, with the backdoor trigger added
    trigger_values = make_trigger(X_train, y_train)
    X_triggered = add_trigger(X_test[y_test == 1], trigger_values)

    # Default baseline-only invocation retains the original output location.
    output = args.output or (
        RESULTS_FILE if args.defences == ["none"] and args.split == "test"
        else Path(f"results/defences_{args.split}.csv")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(model_name, seed, labels, attack, rate, defence, features=None):
        if defence == "spectral" and model_name != "mlp":
            return  # spectral signatures only applies to the MLP
        print(f"Running {model_name}, {attack}, {rate:.0%}, {defence}, seed {seed}...", flush=True)
        row = run_one(
            model_name, seed, X_train if features is None else features, labels, X_test, y_test,
            attack=attack, rate=rate, defence=defence, attack_type_test=at_test,
            k=args.k, n_partitions=args.partitions, epochs=args.epochs,
            X_triggered=X_triggered,
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
                if attack_name == "backdoor":
                    # Trigger added to some training attack flows, which are relabelled benign
                    features, poisoned, _ = apply_backdoor(
                        X_train, y_train, rate, TRIGGER_FEATURES, trigger_values, seed=seed)
                else:
                    features = None
                    poisoned, _ = ATTACKS[attack_name](y_train, rate, seed=seed)
                poisoned = pd.Series(poisoned, index=y_train.index)
                for model_name in args.models:
                    for defence in dict.fromkeys(args.defences):
                        record(model_name, seed, poisoned, attack_name, rate, defence, features)

    results = pd.DataFrame(rows)
    summary = results.groupby(["model", "attack", "rate", "defence"])[
        ["precision", "recall", "f1", "fpr", "asr_backdoor"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(output.with_name(output.stem + "_summary.csv"))
    print(summary.round(4))
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()