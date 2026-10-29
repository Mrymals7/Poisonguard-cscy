import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.attacks.hopskipjump import hopskipjump_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_random_forest(X_train, y_train)

# Select reproducible malicious validation samples.
malicious = X_val.loc[y_val == 1]

X_attack = malicious.sample(
    n=min(50, len(malicious)),
    random_state=42
)
y_attack = y_val.loc[X_attack.index]

# Preliminary feature constraints.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

before = model.predict(X_attack)

X_adv = hopskipjump_attack(
    model,
    X_attack,
    allowed_features=allowed_features
)

X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

after = model.predict(X_adv)

print("Before HopSkipJump:", recall_score(
    y_attack, before, zero_division=0
))
print("After HopSkipJump:", recall_score(
    y_attack, after, zero_division=0
))
print("Changed values:", np.count_nonzero(
    X_adv.to_numpy() - X_attack.to_numpy()
))

# Check that fixed features were preserved.
fixed_features = [
    c for c in X_attack.columns if c not in allowed_features
]

assert np.array_equal(
    X_adv[fixed_features].to_numpy(dtype=np.float32),
    X_attack[fixed_features].to_numpy(dtype=np.float32)
), "HopSkipJump modified a restricted feature"

print("Feature constraints verified.")
