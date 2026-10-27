import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.data import load_processed
from src.defences.knn_sanitisation import sanitise_labels
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
    print("Trial training samples:", len(y_small))
    start = time.perf_counter()
    cleaned_y = sanitise_labels(X_small, poisoned_labels, k=5)
    sanitisation_seconds = time.perf_counter() - start

    changed = np.count_nonzero(cleaned_y != np.asarray(poisoned_labels))
    print("Labels changed:", changed)
    print(f"Sanitisation time: {sanitisation_seconds:.2f} seconds")
    for name, labels in [("Without defence", poisoned_labels),
                         ("With defence", cleaned_y)]:
        model = train_random_forest(X_small, labels, seed=42)
        pred = model.predict(X_val)
        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()

        print(f"\n{name}")
        print(f"Precision: {precision_score(y_val, pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_val, pred, zero_division=0):.4f}")
        print(f"F1: {f1_score(y_val, pred, zero_division=0):.4f}")
        print(f"False-positive rate: {fp / (fp + tn):.4f}")

