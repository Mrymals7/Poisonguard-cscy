from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 score: {f1_score(y_true, y_pred):.4f}")
    print(f"FPR: {fp / (fp + tn):.4f}")


X_train, y_train, _ = load_processed("train")
X_test, y_test, _ = load_processed("test")

mlp = train_mlp(X_train, y_train)
rf = train_random_forest(X_train, y_train)

evaluate("MLP - Clean Test", y_test, predict_mlp(mlp, X_test))
evaluate("RF - Clean Test", y_test, rf.predict(X_test))
print("\nClean baseline evaluation completed.")

import pandas as pd

X_attack = X_test[y_test == 1].head(100)
y_attack = y_test.loc[X_attack.index]

X_fgsm = fgsm_attack(mlp, X_attack, y_attack, epsilon=0.01)
X_pgd = pgd_attack(mlp, X_attack, y_attack, epsilon=0.01)

X_fgsm = pd.DataFrame(X_fgsm, columns=X_attack.columns, index=X_attack.index)
X_pgd = pd.DataFrame(X_pgd, columns=X_attack.columns, index=X_attack.index)

print("\nPreliminary attack evaluation")
print("Before:", recall_score(y_attack, predict_mlp(mlp, X_attack)))
print("After FGSM:", recall_score(y_attack, predict_mlp(mlp, X_fgsm)))
print("After PGD:", recall_score(y_attack, predict_mlp(mlp, X_pgd)))
defended = adversarial_train_mlp(X_train, y_train)

evaluate(
    "Defended MLP - Clean Test",
    y_test,
    predict_mlp(defended, X_test)
)

X_adv_defended = fgsm_attack(
    defended, X_attack, y_attack, epsilon=0.01
)

X_adv_defended = pd.DataFrame(
    X_adv_defended,
    columns=X_attack.columns,
    index=X_attack.index
)

print("\nPreliminary defended MLP evaluation")
print(
    "Defended before attack:",
    recall_score(y_attack, predict_mlp(defended, X_attack))
)
print(
    "Defended after direct FGSM:",
    recall_score(y_attack, predict_mlp(defended, X_adv_defended))
)