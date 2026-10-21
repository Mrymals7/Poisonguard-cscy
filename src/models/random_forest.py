from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X, y, seed=42):
    """Train the Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    import time
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    from src.data import load_processed

    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")

    start = time.time()
    model = train_random_forest(X_train, y_train)
    print(f"Training took {time.time() - start:.1f} seconds")

    pred = model.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    print(f"Precision: {precision_score(y_val, pred):.4f}")
    print(f"Recall:    {recall_score(y_val, pred):.4f}")
    print(f"F1 score:  {f1_score(y_val, pred):.4f}")
    print(f"False-positive rate: {fp / (fp + tn):.4f}")