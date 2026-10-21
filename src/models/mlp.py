import torch
from torch import nn


class MLP(nn.Module):
    """A small neural network: 80 features -> 128 -> 64 -> 2 outputs (benign, attack)."""

    def __init__(self, n_features=80):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, 2)

    def hidden(self, x):
        """Return the 64-neuron hidden layer (used by spectral signatures)."""
        return self.body(x)

    def forward(self, x):
        """Return the scores for benign and attack."""
        return self.head(self.body(x))

def train_mlp(X, y, seed=42, epochs=10, batch_size=256, lr=1e-3):
    """Train the MLP baseline on features X and labels y."""
    torch.manual_seed(seed)

    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.long)

    # Give attacks more weight because they are rarer (like class_weight="balanced")
    counts = torch.bincount(y_t)
    weights = len(y_t) / (2 * counts.float())

    model = MLP(n_features=X_t.shape[1])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        print(f"Epoch {epoch + 1}/{epochs} - loss {total / len(X_t):.4f}")

    model.eval()
    return model


def predict_mlp(model, X):
    """Return 0/1 predictions for features X."""
    with torch.no_grad():
        scores = model(torch.tensor(X.values, dtype=torch.float32))
    return scores.argmax(dim=1).numpy()

if __name__ == "__main__":
    import time
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    from src.data import load_processed

    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")

    start = time.time()
    model = train_mlp(X_train, y_train)
    print(f"Training took {time.time() - start:.1f} seconds")

    pred = predict_mlp(model, X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    print(f"Precision: {precision_score(y_val, pred):.4f}")
    print(f"Recall:    {recall_score(y_val, pred):.4f}")
    print(f"F1 score:  {f1_score(y_val, pred):.4f}")
    print(f"False-positive rate: {fp / (fp + tn):.4f}")