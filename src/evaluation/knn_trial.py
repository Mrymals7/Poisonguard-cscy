import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import evaluate


from src.data import load_processed
from src.defences.knn_sanitisation import sanitise_labels
from src.models.random_forest import train_random_forest
from src.models.mlp import train_mlp, predict_mlp
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
    best_k = 5
    best_f1 = -1.0
    for k in [3, 5, 7, 11]:
        candidate_y = sanitise_labels(X_small, poisoned_labels, k=k)
        candidate_model = train_random_forest(X_small, candidate_y, seed=42)
        candidate_pred = candidate_model.predict(X_val)
        candidate_f1 = evaluate(y_val, candidate_pred)["f1"]
        print(f"k={k}: validation F1={candidate_f1:.4f}")
        if candidate_f1 > best_f1:
            best_k = k
            best_f1 = candidate_f1
    print(f"Selected k: {best_k}")
    start = time.perf_counter()
    cleaned_y = sanitise_labels(X_small, poisoned_labels, k=best_k)
    sanitisation_seconds = time.perf_counter() - start

    changed = np.count_nonzero(cleaned_y != np.asarray(poisoned_labels))
    print("Labels changed:", changed)
    print(f"Sanitisation time: {sanitisation_seconds:.2f} seconds")
    for name, labels in [("Without defence", poisoned_labels),
                         ("With defence", cleaned_y)]:
        model = train_random_forest(X_small, labels, seed=42)
        pred = model.predict(X_val)

        print(f"\n{name}")
        scores = evaluate(y_val, pred)
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
        model = train_mlp(X_small, pd.Series(np.asarray(labels), index=X_small.index), seed=42)
        pred = predict_mlp(model, X_val)
        print(f"\nMLP - {name}")
        scores = evaluate(y_val, pred)
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
