import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

model = train_mlp(X_train, y_train)

X_attack = X_val[y_val == 1].head(100)
y_attack = y_val.loc[X_attack.index]

before = predict_mlp(model, X_attack)

X_fgsm = fgsm_attack(model, X_attack, y_attack, epsilon=0.01)
X_pgd = pgd_attack(model, X_attack, y_attack, epsilon=0.01)

X_fgsm = pd.DataFrame(X_fgsm, columns=X_attack.columns)
X_pgd = pd.DataFrame(X_pgd, columns=X_attack.columns)

fgsm_pred = predict_mlp(model, X_fgsm)
pgd_pred = predict_mlp(model, X_pgd)

print("Before attack:", recall_score(y_attack, before))
print("After FGSM:", recall_score(y_attack, fgsm_pred))
print("After PGD:", recall_score(y_attack, pgd_pred))
