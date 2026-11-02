from pathlib import Path

import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defences.adversarial_training import adversarial_train_mlp


SEED = 42
N_SAMPLES = 10000
EPSILONS = [0.01, 0.03, 0.05]

# Load training and validation data.
X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

# Restrict attacks to forward-direction features.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

# Select reproducible malicious validation samples.
malicious = X_val.loc[y_val == 1]

if malicious.empty:
    raise ValueError("No malicious validation samples found")

X_attack = malicious.sample(
    n=min(100, len(malicious)),
    random_state=SEED
)
y_attack = y_val.loc[X_attack.index]

ATTACKS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}

results = []


def evaluate_model(
    model,
    model_name,
    training_attack,
    training_epsilon
):
    """Evaluate clean recall and direct FGSM/PGD attacks."""

    clean_predictions = predict_mlp(model, X_attack)

    clean_recall = recall_score(
        y_attack,
        clean_predictions,
        zero_division=0
    )

    results.append({
        "model": model_name,
        "training_attack": training_attack,
        "training_epsilon": training_epsilon,
        "seed": SEED,
        "evaluation_attack": "clean",
        "attack_epsilon": 0.0,
        "recall": clean_recall,
    })

    print(f"\n{model_name}")
    print(f"Clean Recall: {clean_recall:.4f}")

    for attack_name, attack_fn in ATTACKS.items():
        for attack_epsilon in EPSILONS:

            X_adv_values = attack_fn(
                model,
                X_attack,
                y_attack,
                epsilon=attack_epsilon,
                allowed_features=allowed_features
            )

            X_adv = pd.DataFrame(
                X_adv_values,
                columns=X_attack.columns,
                index=X_attack.index
            )

            predictions = predict_mlp(model, X_adv)

            recall = recall_score(
                y_attack,
                predictions,
                zero_division=0
            )

            results.append({
                "model": model_name,
                "training_attack": training_attack,
                "training_epsilon": training_epsilon,
                "seed": SEED,
                "evaluation_attack": attack_name,
                "attack_epsilon": attack_epsilon,
                "recall": recall,
            })

            print(
                f"{attack_name.upper()} "
                f"(epsilon={attack_epsilon}): Recall={recall:.4f}"
            )


# Train and evaluate the undefended baseline.
print("\n========== BASELINE MLP ==========")

baseline = train_mlp(
    X_train,
    y_train,
    seed=SEED
)

evaluate_model(
    baseline,
    "Baseline MLP",
    "none",
    0.0
)


# Compare FGSM and PGD adversarial training
# using three training epsilon values.
for training_attack in ["fgsm", "pgd"]:

    for training_epsilon in EPSILONS:

        print("\n====================================")
        print(
            f"Training defence: {training_attack.upper()} | "
            f"Epsilon: {training_epsilon}"
        )
        print("====================================")

        defended = adversarial_train_mlp(
            X_train,
            y_train,
            epsilon=training_epsilon,
            n_samples=N_SAMPLES,
            allowed_features=allowed_features,
            attack_type=training_attack,
            seed=SEED
        )

        model_name = (
            f"Defended MLP - {training_attack.upper()} "
            f"- epsilon {training_epsilon}"
        )

        evaluate_model(
            defended,
            model_name,
            training_attack,
            training_epsilon
        )


# Save all comparison results.
Path("results").mkdir(exist_ok=True)

results_path = Path(
    "results/mlp_defence_epsilon_sweep.csv"
)

pd.DataFrame(results).to_csv(
    results_path,
    index=False
)

print("\n========== EXPERIMENT COMPLETED ==========")
print(f"Results saved to: {results_path}")
print(f"Total evaluation records: {len(results)}")
