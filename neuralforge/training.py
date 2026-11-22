"""Mini-batch SGD and Adam. Histories include the untrained epoch zero."""
import numpy as np


def batches(n, size, epochs, seed):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        indices = rng.permutation(n)
        yield [indices[i:i + size] for i in range(0, n, size)]


class Optimizer:
    def __init__(self, layers, method, lr):
        if method not in {"sgd", "adam"} or lr <= 0:
            raise ValueError("Use sgd or adam with a positive learning rate.")
        self.method, self.lr, self.step_number = method, lr, 0
        self.params = [p for layer in layers for p in layer]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def update(self, gradients):
        self.step_number += 1
        grads = [g for layer in gradients for g in layer]
        for i, (param, grad) in enumerate(zip(self.params, grads, strict=True)):
            if self.method == "adam":
                self.m[i] = 0.9 * self.m[i] + 0.1 * grad
                self.v[i] = 0.999 * self.v[i] + 0.001 * grad**2
                mean = self.m[i] / (1 - 0.9**self.step_number)
                variance = self.v[i] / (1 - 0.999**self.step_number)
                grad = mean / (np.sqrt(variance) + 1e-8)
            param -= self.lr * grad


def fit(model, train, validation, *, epochs=100, batch_size=64, method="adam", lr=0.01, seed=7):
    x, y = train
    optimizer = Optimizer(model.layers, method, lr)

    def measure():
        return {"train": model.metrics(x, y), "validation": model.metrics(*validation)}

    history = [measure()]
    for epoch in batches(len(y), batch_size, epochs, seed):
        for idx in epoch:
            _, gradients = model.loss_gradients(x[idx], y[idx])
            optimizer.update(gradients)
        history.append(measure())
    return history
