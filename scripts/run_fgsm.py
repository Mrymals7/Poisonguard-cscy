import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_mlp(X_train, y_train)

# Select random malicious validation examples.
malicious = X_val.loc[y_val == 1]

X_attack = malicious.sample(
    n=min(100, len(malicious)),
    random_state=42
)
y_attack = y_val.loc[X_attack.index]

# Preliminary restrictions on modifiable features.
allowed_features = [
    col for col in X_train.columns
    if col.startswith("Fwd")
]

if not allowed_features:
    raise ValueError("No allowed Fwd features found")

before = predict_mlp(model, X_attack)

X_adv = fgsm_attack(
    model,
    X_attack,
    y_attack,
    epsilon=0.01,
    allowed_features=allowed_features
)

X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

after = predict_mlp(model, X_adv)

print("Before FGSM:", recall_score(y_attack, before))
print("After FGSM:", recall_score(y_attack, after))
