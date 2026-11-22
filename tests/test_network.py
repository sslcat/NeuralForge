import numpy as np
import pytest

from neuralforge.check import check_gradients
from neuralforge.data import spirals
from neuralforge.network import MLP, cross_entropy
from neuralforge.training import fit


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_shapes_and_gradients(activation):
    x = np.random.default_rng(9).normal(size=(7, 2))
    y = np.arange(len(x)) % 3
    model = MLP([2, 5, 4, 3], activation)
    # Centered differences require a smooth neighborhood. An all-zero
    # hidden row and zero biases can place the next ReLU exactly at its kink.
    if activation == "relu":
        for _, bias in model.layers[:-1]:
            bias[:] = np.linspace(0.12, 0.27, len(bias))
    logits, cache = model.forward(x)
    assert logits.shape == (7, 3)
    assert [a.shape for a in cache] == [(7, 2), (7, 5), (7, 4)]
    _, grads = model.loss_gradients(x, y)
    for (w, b), (dw, db) in zip(model.layers, grads, strict=True):
        assert (w.shape, b.shape) == (dw.shape, db.shape)
    error = check_gradients(model, x, y)
    assert error["relative_l2"] < 1e-7
    assert error["max_absolute"] < 1e-7


def test_extreme_logits():
    logits = np.array([[10000., -10000.], [-10000., 10000.]])
    loss, delta = cross_entropy(logits, np.array([1, 0]))
    assert loss == 20000
    np.testing.assert_allclose(delta, [[0.5, -0.5], [-0.5, 0.5]])
    shifted, _ = cross_entropy(logits + 30000, np.array([1, 0]))
    assert shifted == loss


def test_data_splits():
    splits = spirals()
    for (x, y), (again, labels), n in zip(splits, spirals(), (600, 200, 200)):
        np.testing.assert_array_equal(x, again)
        np.testing.assert_array_equal(y, labels)
        assert x.shape == (n, 2)
        assert y.sum() == n // 2
    all_points = np.concatenate([x for x, y in splits])
    assert len(np.unique(all_points, axis=0)) == len(all_points)
    np.testing.assert_allclose(splits[0][0].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(splits[0][0].std(0), 1)


@pytest.mark.parametrize("method", ["sgd", "adam"])
def test_deterministic_training(method):
    train, val, _ = spirals(per_class=50)
    models = [MLP([2, 8, 2]) for _ in range(2)]
    histories = [fit(m, train, val, epochs=4, batch_size=17, method=method) for m in models]
    assert histories[0] == histories[1]
    assert histories[0][-1]["train"]["loss"] < histories[0][0]["train"]["loss"]


def test_held_out_learning():
    train, val, test = spirals()
    model = MLP([2, 32, 32, 2])
    before = model.metrics(*test)
    fit(model, train, val, epochs=50)
    after = model.metrics(*test)
    assert after["accuracy"] > max(0.9, before["accuracy"] + 0.2)
    assert after["loss"] < before["loss"] / 2
