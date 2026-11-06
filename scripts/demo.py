"""PoisonGuard live demo: a backdoor attack on the MLP and a defence that stops it.

Run from the repository folder (after `bash setup.sh`):
    python -m scripts.demo              # about 1 minute
    python -m scripts.demo --with-knn   # also shows k-NN sanitisation (about 3 more minutes)
"""
import argparse

import pandas as pd

from src.data import load_processed
from src.attacks.backdoor import apply_backdoor
from src.run_experiments import TRIGGER_FEATURES, make_trigger, add_trigger, run_one

SEED = 1        # seed 1 gives a clear example; results.csv has seeds 1-3
RATE = 0.01     # the attacker poisons 1% of the training rows


def show(title, row):
    """Print the numbers that matter for one model."""
    print(f"\n--- {title} ---")
    print(f"Normal traffic  : F1 {row['f1']:.3f}, recall {row['recall']:.3f}, "
          f"false alarms {row['fpr']:.2%}")
    print(f"Triggered attacks classified benign (backdoor success): {row['asr_backdoor']:.1%}")
    print(f"Time            : {row['fit_total_time_s']:.0f} s")


def main():
    parser = argparse.ArgumentParser(description="PoisonGuard live demo")
    parser.add_argument("--with-knn", action="store_true", help="also run k-NN sanitisation")
    args = parser.parse_args()

    print("=== PoisonGuard live demo: backdoor attack on the MLP intrusion detector ===")
    X_train, y_train, _ = load_processed("train")
    X_test, y_test, at_test = load_processed("test")

    trigger_values = make_trigger(X_train, y_train)
    X_triggered = add_trigger(X_test[y_test == 1], trigger_values)
    print(f"\nTraining flows: {len(X_train):,}   Test flows: {len(X_test):,}")
    print("Backdoor trigger: " + ", ".join(
        f"{feature} = {value:.4f}" for feature, value in zip(TRIGGER_FEATURES, trigger_values)))
    print(f"Test attack flows with the trigger added: {len(X_triggered):,}")

    common = dict(attack_type_test=at_test, X_triggered=X_triggered)

    # 1. Clean model: the trigger alone does not fool it
    clean = run_one("mlp", SEED, X_train, y_train, X_test, y_test, **common)
    show("1. Clean MLP (no attack)", clean)

    # 2. Attacker adds the trigger to 1% of training rows (attack flows relabelled benign)
    X_poisoned, y_poisoned, poisoned_rows = apply_backdoor(
        X_train, y_train, RATE, TRIGGER_FEATURES, trigger_values, seed=SEED)
    y_poisoned = pd.Series(y_poisoned, index=X_poisoned.index)
    print(f"\nAttacker poisons {len(poisoned_rows):,} training flows ({RATE:.0%}): "
          "attack flows with the trigger, labelled benign")
    attacked = run_one("mlp", SEED, X_poisoned, y_poisoned, X_test, y_test,
                       attack="backdoor", rate=RATE, **common)
    show("2. Backdoored MLP (no defence)", attacked)
    print("   -> Normal accuracy looks fine, so normal testing would not notice the attack.")

    # 3. Defence: partition ensemble (3 MLPs on separate parts of the data, majority vote)
    defended = run_one("mlp", SEED, X_poisoned, y_poisoned, X_test, y_test,
                       attack="backdoor", rate=RATE, defence="partition", **common)
    show("3. Defended: partition ensemble", defended)

    if args.with_knn:
        knn = run_one("mlp", SEED, X_poisoned, y_poisoned, X_test, y_test,
                      attack="backdoor", rate=RATE, defence="knn", **common)
        show("4. Defended: k-NN label sanitisation", knn)

    print("\n=== Summary: backdoor success (target: below 10%) ===")
    print(f"Clean model        : {clean['asr_backdoor']:.1%}")
    print(f"Backdoored model   : {attacked['asr_backdoor']:.1%}")
    print(f"Partition ensemble : {defended['asr_backdoor']:.1%}")
    if args.with_knn:
        print(f"k-NN sanitisation  : {knn['asr_backdoor']:.1%}")


if __name__ == "__main__":
    main()