import torch
from torch import nn


def pgd_attack(model, X, y, epsilon=0.03, alpha=0.005, steps=10, allowed_features=None):
    """Generate adversarial examples using PGD."""
    model.eval()

    x = torch.tensor(X.to_numpy(copy=True), dtype=torch.float32)
    labels = torch.tensor(y.to_numpy(copy=True), dtype=torch.long)

    x_adv = x.clone().detach()

    loss_fn = nn.CrossEntropyLoss()

    if allowed_features is not None:
        mask = torch.zeros_like(x)
        for feature in allowed_features:
            index = X.columns.get_loc(feature)
            mask[:, index] = 1.0
    else:
        mask = torch.ones_like(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        loss = loss_fn(model(x_adv), labels)
        gradient = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * gradient.sign() * mask
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach().numpy()