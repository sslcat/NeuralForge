from copy import deepcopy

import pytest

from neuralforge.data import spirals
from neuralforge.network import MLP
from neuralforge.training import fit


@pytest.mark.parametrize("method", ["adam", "sgd"])
def test_reference_matches_logits(method):
    pytest.importorskip("torch")
    from neuralforge.reference import compare
    train, val, test = spirals(per_class=30)
    initial = MLP([2, 8, 2])
    trained = deepcopy(initial)
    options = dict(method=method, lr=0.01, epochs=3, batch_size=11, seed=7)
    fit(trained, train, val, **options)
    assert compare(initial, trained, train, test, **options)["max_logit_difference"] < 1e-10
