import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_mlp(X_train, y_train)

X_attack = X_val[y_val == 1].head(100)
y_attack = y_val.loc[X_attack.index]

before = predict_mlp(model, X_attack)

X_adv = fgsm_attack(model, X_attack, y_attack)
X_adv = pd.DataFrame(X_adv, columns=X_attack.columns)

after = predict_mlp(model, X_adv)

print("Before FGSM:", recall_score(y_attack, before))
print("After FGSM:", recall_score(y_attack, after))