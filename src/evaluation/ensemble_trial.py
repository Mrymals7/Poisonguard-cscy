import pandas as pd
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import evaluate
from src.data import load_processed
from src.defences.partition_ensemble import train_partition_ensemble, predict_partition_ensemble
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
    poisoned_labels = pd.Series(poisoned_labels, index=y_small.index)
    model = train_random_forest(X_small, poisoned_labels, seed=42)
    pred = model.predict(X_val)
    print("Single Random Forest")
    scores = evaluate(y_val, pred)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    models = train_partition_ensemble(X_small, poisoned_labels)
    pred = predict_partition_ensemble(models, X_val)
    print("Partition Ensemble")
    scores = evaluate(y_val, pred)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    model = train_mlp(X_small, poisoned_labels, seed=42)
    pred = predict_mlp(model, X_val)
    



    print("Single MLP")
    scores = evaluate(y_val, pred)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    models = train_partition_ensemble(X_small, poisoned_labels, model_type="mlp")
    pred = predict_partition_ensemble(models, X_val, model_type="mlp")
    print("MLP Partition Ensemble")
    scores = evaluate(y_val, pred)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
