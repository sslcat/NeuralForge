"""Dense layers, mean cross-entropy and reverse-mode differentiation."""
import numpy as np


def cross_entropy(logits, labels):
    """Return mean loss and its derivative with respect to each logit."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = exponentials.sum(axis=1, keepdims=True)
    rows = np.arange(len(labels))
    loss = (np.log(denominator[:, 0]) - shifted[rows, labels]).mean()
    delta = exponentials / denominator
    delta[rows, labels] -= 1
    return float(loss), delta / len(labels)


class MLP:
    def __init__(self, widths, activation="tanh", seed=17):
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError("Provide positive input and output widths.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Choose relu or tanh.")
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.layers = []
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            gain = 2 if activation == "relu" and index < len(widths) - 2 else 1
            weights = rng.normal(size=(inputs, outputs)) * np.sqrt(gain / inputs)
            self.layers.append((weights, np.zeros(outputs)))

    def forward(self, x):
        activations = [x]
        for weights, bias in self.layers[:-1]:
            z = activations[-1] @ weights + bias
            activations.append(np.tanh(z) if self.activation == "tanh" else np.maximum(z, 0))
        weights, bias = self.layers[-1]
        return activations[-1] @ weights + bias, activations

    def loss_gradients(self, x, y):
        logits, activations = self.forward(x)
        loss, delta = cross_entropy(logits, y)
        gradients = []
        for i in reversed(range(len(self.layers))):
            a = activations[i]
            gradients.append((a.T @ delta, delta.sum(axis=0)))
            if i:
                slope = 1 - a**2 if self.activation == "tanh" else (a > 0)
                delta = (delta @ self.layers[i][0].T) * slope
        return loss, gradients[::-1]

    def metrics(self, x, y):
        logits, _ = self.forward(x)
        return {"loss": cross_entropy(logits, y)[0],
                "accuracy": float((logits.argmax(axis=1) == y).mean())}
