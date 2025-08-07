import numpy as np


def uniform_selections(size, split_fractions, seed=None):
    assert np.sum(split_fractions) <= 1

    indices = np.arange(0, size)

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices, axis=0)

    split_sizes = list()

    for split_fraction in split_fractions:
        size = int(split_fraction * indices.size)
        split_sizes.append(size)

    split_indices = np.cumsum(split_sizes).tolist()

    indices = np.split(indices, split_indices)

    return indices
