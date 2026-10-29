import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.random_forest import train_random_forest
from src.attacks.hopskipjump import hopskipjump_attack


CACHE_PATH = Path("results/cache/hopskipjump_val.npz")

ATTACK_SETTINGS = {
    "attack": "HopSkipJump",
    "targeted": False,
    "max_iter": 10,
    "max_eval": 100,
    "init_eval": 10,
    "sample_size": 50,
    "sample_seed": 42,
    "feature_rule": "Fwd prefix",
}


def fingerprint(X, model, allowed_features):
    """Identify the input samples, model and attack settings."""
    digest = hashlib.sha256()

    digest.update(
        json.dumps(ATTACK_SETTINGS, sort_keys=True).encode()
    )
    digest.update(
        json.dumps(list(X.columns)).encode()
    )
    digest.update(
        json.dumps(allowed_features).encode()
    )
    digest.update(
        np.ascontiguousarray(
            X.to_numpy(dtype=np.float32)
        ).tobytes()
    )

    # Include the fitted Random Forest structure.
    for tree in model.estimators_:
        digest.update(tree.tree_.feature.tobytes())
        digest.update(tree.tree_.threshold.tobytes())
        digest.update(tree.tree_.value.tobytes())

    return digest.hexdigest()


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_random_forest(X_train, y_train)

malicious = X_val.loc[y_val == 1]

X_attack = malicious.sample(
    n=min(50, len(malicious)),
    random_state=42
)
y_attack = y_val.loc[X_attack.index]

allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

before = model.predict(X_attack)

cache_key = fingerprint(
    X_attack, model, allowed_features
)

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Reuse cached examples only if settings and inputs match.
if CACHE_PATH.exists():
    with np.load(CACHE_PATH, allow_pickle=False) as cached:
        stored_key = str(cached["fingerprint"].item())

        if stored_key == cache_key:
            X_adv_values = cached["examples"].copy()

            if X_adv_values.shape != X_attack.shape:
                raise ValueError("Cached examples have an invalid shape")

            print("Using cached HopSkipJump examples.")
        else:
            X_adv_values = None
else:
    X_adv_values = None

if X_adv_values is None:
    print("Generating new HopSkipJump examples...")

    X_adv_values = hopskipjump_attack(
        model,
        X_attack,
        allowed_features=allowed_features
    )

    np.savez_compressed(
        CACHE_PATH,
        examples=np.asarray(X_adv_values),
        fingerprint=np.array(cache_key)
    )

    print("HopSkipJump examples saved to cache.")

X_adv = pd.DataFrame(
    X_adv_values,
    columns=X_attack.columns,
    index=X_attack.index
)

fixed_features = [
    c for c in X_attack.columns
    if c not in allowed_features
]

assert np.array_equal(
    X_adv[fixed_features].to_numpy(dtype=np.float32),
    X_attack[fixed_features].to_numpy(dtype=np.float32)
), "HopSkipJump modified a restricted feature"

after = model.predict(X_adv)

print(
    "Before HopSkipJump:",
    recall_score(y_attack, before, zero_division=0)
)
print(
    "After HopSkipJump:",
    recall_score(y_attack, after, zero_division=0)
)
print(
    "Changed values:",
    np.count_nonzero(
        X_adv.to_numpy() - X_attack.to_numpy()
    )
)
print("Feature constraints verified.")
