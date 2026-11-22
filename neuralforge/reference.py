"""Optional float64 CPU reference, initialized directly from the NumPy model."""
import numpy as np
import torch
from torch import nn

from .training import batches


def compare(initial, trained, train, test, *, epochs, batch_size, seed, lr, method):
    torch.set_num_threads(1)
    modules = []
    for i, (weights, bias) in enumerate(initial.layers):
        linear = nn.Linear(*weights.shape, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(weights.T))
            linear.bias.copy_(torch.from_numpy(bias))
        modules.append(linear)
        if i < len(initial.layers) - 1:
            modules.append(nn.Tanh() if initial.activation == "tanh" else nn.ReLU())
    model = nn.Sequential(*modules)
    x, y = torch.from_numpy(train[0]), torch.from_numpy(train[1]).long()
    if method == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in batches(len(y), batch_size, epochs, seed):
        for idx in epoch:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(test[0]))
        labels = torch.from_numpy(test[1]).long()
        numpy_logits, _ = trained.forward(test[0])
        return {"loss": nn.functional.cross_entropy(logits, labels).item(),
                "accuracy": (logits.argmax(1) == labels).double().mean().item(),
                "max_logit_difference": float(np.max(np.abs(numpy_logits - logits.numpy()))),
                "torch_version": str(torch.__version__)}
