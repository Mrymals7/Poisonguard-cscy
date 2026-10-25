import pandas as pd
from sklearn.metrics import recall_score

from src.data import load_processed
from src.models.mlp import train_mlp, predict_mlp
from src.attacks.fgsm import fgsm_attack
from src.defences.adversarial_training import adversarial_train_mlp


X_train, y_train, _ = load_processed("train")
X_val, y_val, _ = load_processed("val")

baseline = train_mlp(X_train, y_train)
defended = adversarial_train_mlp(X_train, y_train)

X_attack = X_val[y_val == 1].head(100)
y_attack = y_val.loc[X_attack.index]

X_adv = fgsm_attack(baseline, X_attack, y_attack)
X_adv = pd.DataFrame(
    X_adv,
    columns=X_attack.columns,
    index=X_attack.index
)

baseline_clean = predict_mlp(baseline, X_attack)
baseline_adv = predict_mlp(baseline, X_adv)
defended_clean = predict_mlp(defended, X_attack)
defended_adv = predict_mlp(defended, X_adv)
X_adv_defended = fgsm_attack(
    defended, X_attack, y_attack, epsilon=0.01
)

X_adv_defended = pd.DataFrame(
    X_adv_defended,
    columns=X_attack.columns,
    index=X_attack.index
)

defended_whitebox = predict_mlp(defended, X_adv_defended)

print("Baseline clean:", recall_score(y_attack, baseline_clean))
print("Baseline attacked:", recall_score(y_attack, baseline_adv))
print("Defended clean:", recall_score(y_attack, defended_clean))
print("Defended attacked:", recall_score(y_attack, defended_adv))
print("Defended under direct FGSM:",
      recall_score(y_attack, defended_whitebox))