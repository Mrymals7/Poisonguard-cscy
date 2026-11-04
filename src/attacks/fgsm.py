import torch
from torch import nn 

def fgsm_attack(model, X, y, epsilon=0.01, allowed_features=None):
    """Generate adversarial examples using FGSM."""
    model.eval()
    x = torch.tensor(X.to_numpy(copy=True), dtype=torch.float32)
    labels = torch.tensor(y.to_numpy(copy=True), dtype=torch.long)

    x.requires_grad_(True)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(model(x),labels)
    gradient = torch.autograd.grad(loss,x)[0]

    if allowed_features is not None:
        mask = torch.zeros_like(gradient)
        for feature in allowed_features:
            index = X.columns.get_loc(feature)
            mask[:, index] = 1.0
        gradient = gradient * mask
        
    x_adv = x + epsilon * gradient.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach().numpy()
