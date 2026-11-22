"""Interleaved noisy spirals and a stratified 60/20/20 split."""
import numpy as np


def spirals(seed=41, per_class=500):
    if per_class < 5:
        raise ValueError("Each arm needs at least five points.")
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for label in (0, 1):
        radius = rng.uniform(0.05, 1, per_class)
        angle = radius * 2 * np.pi + label * np.pi + rng.normal(0, 0.16, per_class)
        points = radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
        points += rng.normal(0, 0.018, points.shape)
        indices = np.split(rng.permutation(per_class), [3 * per_class // 5, 4 * per_class // 5])
        for split, subset in zip(splits, indices):
            split.append((points[subset], np.full(len(subset), label)))
    result = []
    for arms in splits:
        x, y = (np.concatenate(parts) for parts in zip(*arms))
        order = rng.permutation(len(y))
        result.append((x[order], y[order]))
    mean, scale = result[0][0].mean(axis=0), result[0][0].std(axis=0)
    return tuple(((x - mean) / scale, y) for x, y in result)
