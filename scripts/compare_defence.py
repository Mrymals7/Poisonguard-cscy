import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defences.adversarial_training import adversarial_train_mlp


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

# Preliminary feature constraints for the experiments.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

# Train the baseline and FGSM-defended models.
baseline = train_mlp(X_train, y_train)

defended = adversarial_train_mlp(
    X_train,
    y_train,
    epsilon=0.01,
    n_samples=10000,
    allowed_features=allowed_features,
    attack_type="fgsm"
)

# Randomly select malicious validation examples.
malicious = X_val.loc[y_val == 1]

X_attack = malicious.sample(
    n=min(100, len(malicious)),
    random_state=42
)
y_attack = y_val.loc[X_attack.index]


def as_dataframe(values):
    return pd.DataFrame(
        values,
        columns=X_attack.columns,
        index=X_attack.index
)


def evaluate(name, model, attack_fn):
    # Direct attack: generated against the model being evaluated.
    X_adv = attack_fn(
        model,
        X_attack,
        y_attack,
        epsilon=0.01,
        allowed_features=allowed_features
    )

    X_adv = as_dataframe(X_adv)

    predictions = predict_mlp(model, X_adv)

    print(
        f"{name}:",
        recall_score(y_attack, predictions, zero_division=0)
    )


baseline_clean = predict_mlp(baseline, X_attack)
defended_clean = predict_mlp(defended, X_attack)

print("=== MLP Defence Comparison ===")
print(
    "Baseline clean:",
    recall_score(y_attack, baseline_clean, zero_division=0)
)
print(
    "Defended clean:",
    recall_score(y_attack, defended_clean, zero_division=0)
)

evaluate("Baseline under FGSM", baseline, fgsm_attack)
evaluate("Defended under FGSM", defended, fgsm_attack)

evaluate("Baseline under PGD", baseline, pgd_attack)
evaluate("Defended under PGD", defended, pgd_attack)
