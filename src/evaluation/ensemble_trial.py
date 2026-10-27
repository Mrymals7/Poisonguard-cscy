import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.data import load_processed
from src.defences.partition_ensemble import train_partition_ensemble, predict_partition_ensemble
from src.models.random_forest import train_random_forest
from src.attacks.label_flipping import random_label_flipping
if __name__ == "__main__":
    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")
    X_small, _, y_small, _ = train_test_split(
            X_train, y_train,
            train_size=5000,
            stratify=y_train,
            random_state=42,
    )
    poisoned_labels, flipped_indices = random_label_flipping(y_small, 0.10, seed=42)
    poisoned_labels = pd.Series(poisoned_labels, index=y_small.index)
    model = train_random_forest(X_small, poisoned_labels, seed=42)
    pred = model.predict(X_val)
    print("Single Random Forest")
    print(f"Precision: {precision_score(y_val, pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_val, pred, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    print(f"False-positive rate: {fp / (fp + tn):.4f}")
    models = train_partition_ensemble(X_small, poisoned_labels)
    pred = predict_partition_ensemble(models, X_val)
    print("Partition Ensemble")
    print(f"Precision: {precision_score(y_val, pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_val, pred, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    print(f"False-positive rate: {fp / (fp + tn):.4f}")

