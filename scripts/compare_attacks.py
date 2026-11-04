import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_mlp(X_train, y_train)

# Select a reproducible random sample of malicious flows.
malicious = X_val.loc[y_val == 1]

X_attack = malicious.sample(
    n=min(100, len(malicious)),
    random_state=42
)
y_attack = y_val.loc[X_attack.index]

# Restrict perturbations to selected forward-flow features.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

before = predict_mlp(model, X_attack)

X_fgsm = fgsm_attack(
    model,
    X_attack,
    y_attack,
    epsilon=0.01,
    allowed_features=allowed_features
)

X_pgd = pgd_attack(
    model,
    X_attack,
    y_attack,
    epsilon=0.01,
    allowed_features=allowed_features
)

X_fgsm = pd.DataFrame(
    X_fgsm,
    columns=X_attack.columns,
    index=X_attack.index
)

X_pgd = pd.DataFrame(
    X_pgd,
    columns=X_attack.columns,
    index=X_attack.index
)

fgsm_pred = predict_mlp(model, X_fgsm)
pgd_pred = predict_mlp(model, X_pgd)

print("Before attack:", recall_score(y_attack, before))
print("After FGSM:", recall_score(y_attack, fgsm_pred))
print("After PGD:", recall_score(y_attack, pgd_pred))
