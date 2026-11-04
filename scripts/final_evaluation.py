import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.models.random_forest import train_random_forest
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defences.adversarial_training import adversarial_train_mlp


def evaluate(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print(
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1 score: {f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"FPR: {fpr:.4f}")


X_train, y_train, _ = load_processed("train")
X_test, y_test, _ = load_processed("test")

allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

mlp = train_mlp(X_train, y_train)
rf = train_random_forest(X_train, y_train)

evaluate("MLP - Clean Test", y_test, predict_mlp(mlp, X_test))
evaluate("RF - Clean Test", y_test, rf.predict(X_test))

print("\nClean baseline evaluation completed.")

# Random malicious test sample.
malicious = X_test.loc[y_test == 1]

X_attack = malicious.sample(
    n=min(100, len(malicious)),
    random_state=42
)
y_attack = y_test.loc[X_attack.index]


def as_dataframe(values):
    return pd.DataFrame(
        values,
        columns=X_attack.columns,
        index=X_attack.index
    )


X_fgsm = as_dataframe(
    fgsm_attack(
        mlp,
        X_attack,
        y_attack,
        epsilon=0.01,
        allowed_features=allowed_features
    )
)

X_pgd = as_dataframe(
    pgd_attack(
        mlp,
        X_attack,
        y_attack,
        epsilon=0.01,
        allowed_features=allowed_features
    )
)

print("\nPreliminary attack evaluation")
print(
    "Before:",
    recall_score(
        y_attack, predict_mlp(mlp, X_attack),
        zero_division=0
    )
)
print(
    "After FGSM:",
    recall_score(
        y_attack, predict_mlp(mlp, X_fgsm),
        zero_division=0
    )
)
print(
    "After PGD:",
    recall_score(
        y_attack, predict_mlp(mlp, X_pgd),
        zero_division=0
    )
)

# Train the defended model on training data only.
defended = adversarial_train_mlp(
    X_train,
    y_train,
    epsilon=0.01,
    n_samples=10000,
    allowed_features=allowed_features,
    attack_type="fgsm"
)

evaluate(
    "Defended MLP - Clean Test",
    y_test,
    predict_mlp(defended, X_test)
)

# Direct white-box attacks against the defended MLP.
X_fgsm_defended = as_dataframe(
    fgsm_attack(
        defended,
        X_attack,
        y_attack,
        epsilon=0.01,
        allowed_features=allowed_features
    )
)

X_pgd_defended = as_dataframe(
    pgd_attack(
        defended,
        X_attack,
        y_attack,
        epsilon=0.01,
        allowed_features=allowed_features
    )
)

print("\nPreliminary defended MLP evaluation")

for name, examples in [
    ("Defended before attack", X_attack),
    ("Defended after direct FGSM", X_fgsm_defended),
    ("Defended after direct PGD", X_pgd_defended),
]:
    recall = recall_score(
        y_attack,
        predict_mlp(defended, examples),
        zero_division=0
    )
    print(f"{name}: {recall:.4f}")
