import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.models.random_forest import train_random_forest
from src.attacks.fgsm import fgsm_attack


SEED = 42
N_SAMPLES = 100
EPSILON = 0.01

X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

# Train both models with a reproducible seed.
mlp = train_mlp(X_train, y_train, seed=SEED)
rf = train_random_forest(X_train, y_train, seed=SEED)

# Randomly select malicious validation samples.
malicious = X_val.loc[y_val == 1]

if malicious.empty:
    raise ValueError("No malicious validation samples found")

X_attack = malicious.sample(
    n=min(N_SAMPLES, len(malicious)),
    random_state=SEED
)
y_attack = y_val.loc[X_attack.index]

# Identify allowed forward-direction features.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

mlp_before = predict_mlp(mlp, X_attack)
rf_before = rf.predict(X_attack)

# Generate FGSM examples against the MLP.
X_adv = fgsm_attack(
    mlp,
    X_attack,
    y_attack,
    epsilon=EPSILON,
    allowed_features=allowed_features
)

X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

# Evaluate the original MLP and transfer attack on RF.
mlp_after = predict_mlp(mlp, X_adv)
rf_after = rf.predict(X_adv)

print("MLP before:", recall_score(y_attack, mlp_before, zero_division=0))
print("MLP after FGSM:", recall_score(y_attack, mlp_after, zero_division=0))
print("RF before:", recall_score(y_attack, rf_before, zero_division=0))
print("RF after transfer:", recall_score(y_attack, rf_after, zero_division=0))

print("Number of attack samples:", len(X_attack))
print("Number of allowed features:", len(allowed_features))
print("Random seed:", SEED)
