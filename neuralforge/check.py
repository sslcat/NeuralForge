"""Exhaustive centered finite differences on a tiny smooth network."""
import numpy as np

from .network import MLP


def check_gradients(model, x, y, epsilon=1e-5):
    _, gradients = model.loss_gradients(x, y)
    analytical, numerical = [], []
    for layer, grads in zip(model.layers, gradients):
        for param, grad in zip(layer, grads):
            for index in np.ndindex(param.shape):
                old = param[index]
                try:
                    param[index] = old + epsilon
                    high = model.metrics(x, y)["loss"]
                    param[index] = old - epsilon
                    low = model.metrics(x, y)["loss"]
                finally:
                    param[index] = old
                numerical.append((high - low) / (2 * epsilon))
                analytical.append(grad[index])
    a, n = np.array(analytical), np.array(numerical)
    return {"parameters": len(a), "max_absolute": float(np.max(np.abs(a - n))),
            "relative_l2": float(np.linalg.norm(a - n) / max(1e-15, np.linalg.norm(a) + np.linalg.norm(n)))}


def demo_check():
    x = np.random.default_rng(12).normal(size=(6, 2))
    return check_gradients(MLP([2, 4, 3, 2]), x, np.arange(6) % 2)
