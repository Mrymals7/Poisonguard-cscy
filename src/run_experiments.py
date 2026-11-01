import time
from pathlib import Path

import pandas as pd

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
            X_triggered=None):
    """Train one model on the given training data, score it on the test set, return one result row."""
    train_fn, predict_fn = MODELS[model_name]

    start = time.time()
    model = train_fn(X_train, y_train, seed=seed)
    train_time = time.time() - start

    pred = predict_fn(model, X_test)
    row = {"model": model_name, "attack": attack, "rate": rate, "defence": defence, "seed": seed}
    row.update(evaluate(y_test, pred))
    row["asr_clean_attacks"] = attack_success_rate(pred[y_test.values == 1])
    if X_triggered is not None:
        row["asr_backdoor"] = attack_success_rate(predict_fn(model, X_triggered))
    row["train_time_s"] = round(train_time, 1)
    if attack_type_test is not None:
        for name, value in per_class_detection(pred, attack_type_test).items():
            row[f"detect_{name}"] = round(value, 4)
    return row


def main():
    X_train, y_train, _ = load_processed("train")
    X_test, y_test, at_test = load_processed("test")

    # Every test attack flow, with the backdoor trigger added
    trigger_values = make_trigger(X_train, y_train)
    X_triggered = add_trigger(X_test[y_test == 1], trigger_values)

    rows = []

    # 1. Clean baselines
    for model_name in MODELS:
        for seed in SEEDS:
            print(f"Running {model_name}, clean, seed {seed}...")
            rows.append(run_one(model_name, seed, X_train, y_train, X_test, y_test,
                                attack_type_test=at_test, X_triggered=X_triggered))

    # 2. Label-flipping attacks (only the training labels are poisoned; the test set stays clean)
    for attack_name, attack_fn in ATTACKS.items():
        for rate in RATES:
            for model_name in MODELS:
                for seed in SEEDS:
                    print(f"Running {model_name}, {attack_name} {rate:.0%}, seed {seed}...")
                    y_poisoned, _ = attack_fn(y_train, rate, seed=seed)
                    y_poisoned = pd.Series(y_poisoned, index=y_train.index)
                    rows.append(run_one(model_name, seed, X_train, y_poisoned, X_test, y_test,
                                        attack=attack_name, rate=rate, attack_type_test=at_test,
                                        X_triggered=X_triggered))

    # 3. Backdoor attack (trigger added to some training attack flows, relabelled benign)
    for rate in RATES:
        for model_name in MODELS:
            for seed in SEEDS:
                print(f"Running {model_name}, backdoor {rate:.0%}, seed {seed}...")
                X_poisoned, y_poisoned, _ = apply_backdoor(
                    X_train, y_train, rate, TRIGGER_FEATURES, trigger_values, seed=seed)
                y_poisoned = pd.Series(y_poisoned, index=X_poisoned.index)
                rows.append(run_one(model_name, seed, X_poisoned, y_poisoned, X_test, y_test,
                                    attack="backdoor", rate=rate, attack_type_test=at_test,
                                    X_triggered=X_triggered))

    results = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)

    summary = results.groupby(["model", "attack", "rate"])[
        ["precision", "recall", "f1", "fpr", "asr_backdoor"]].agg(["mean", "std"])
    print(summary.round(4))


if __name__ == "__main__":
    main()