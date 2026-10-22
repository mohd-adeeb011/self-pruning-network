"""Classification data; preprocessing uses training rows only."""
import numpy as np


def make_data(seed=2026, examples=1200, noise=0.12):
    if examples < 100 or examples % 2:
        raise ValueError("Use an even dataset size of at least 100.")
    rng = np.random.default_rng(seed)
    n = examples // 2
    angles = rng.uniform(0, np.pi, n)
    first = np.column_stack((np.cos(angles), np.sin(angles)))
    angles = rng.uniform(0, np.pi, n)
    second = np.column_stack((1-np.cos(angles), 0.5-np.sin(angles)))
    x = np.concatenate((first, second)) + rng.normal(0, noise, (examples, 2))
    y = np.repeat([0, 1], n)
    splits = [[], [], []]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        boundaries = [int(0.6*n), int(0.8*n)]
        for bucket, subset in zip(splits, np.split(indices, boundaries)):
            bucket.extend(subset)
    indices = [rng.permutation(bucket) for bucket in splits]
    train_x = x[indices[0]]
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.maximum(scale, 1e-12)
    data = {name: ((x[idx]-mean)/scale, y[idx])
            for name, idx in zip(('train','val','test'), indices)}
    data['normalization'] = (mean, scale)
    return data


def load_data(dataset='moons', seed=2026, examples=1200):
    if dataset == 'moons':
        return make_data(seed, examples)
    if dataset == 'covertype':
        from .covertype_data import load_covertype
        return load_covertype(seed)
    raise ValueError(f'Unknown dataset: {dataset}')
