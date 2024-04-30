import numpy as np


def poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None):
        rng = np.random.RandomState(freezing_seed)
    else:
        rng = np.random.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)  # noqa

    return spikes
