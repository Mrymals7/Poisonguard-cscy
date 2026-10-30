import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate(y_true, y_pred):
    """Return precision, recall, F1 and false-positive rate for binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def attack_success_rate(y_pred):
    """Share of attack flows the model classified as benign (0)."""
    return float((y_pred == 0).mean())

def per_class_detection(y_pred, attack_type):
    """Share of flows flagged as attack (1) for each attack type."""
    table = pd.DataFrame({"attack_type": attack_type.values, "flagged": y_pred == 1})
    return table.groupby("attack_type")["flagged"].mean()