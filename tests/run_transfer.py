import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.models.random_forest import train_random_forest
from src.attacks.fgsm import fgsm_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

mlp = train_mlp(X_train, y_train)
rf = train_random_forest(X_train, y_train)

X_attack = X_val[y_val == 1].head(100)
y_attack = y_val.loc[X_attack.index]

mlp_before = predict_mlp(mlp, X_attack)
rf_before = rf.predict(X_attack)

X_adv = fgsm_attack(mlp, X_attack, y_attack, epsilon=0.01)
X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

mlp_after = predict_mlp(mlp, X_adv)
rf_after = rf.predict(X_adv)

print("MLP before:", recall_score(y_attack, mlp_before))
print("MLP after FGSM:", recall_score(y_attack, mlp_after))
print("RF before:", recall_score(y_attack, rf_before))
print("RF after transfer:", recall_score(y_attack, rf_after))
