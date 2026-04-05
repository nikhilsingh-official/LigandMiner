import math

import numpy as np

def gaussian_nll(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    z = (x - mu) / sigma
    return 0.5 * z * z

def nll_to_unit_interval(nll_total: float, tau: float) -> float:
    if not math.isfinite(nll_total):
        return 1.0
    return 1.0 - math.exp(-nll_total / tau)


def gaussian_nll_vectorized(observed, means, stds):
    observed = np.asarray(observed, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.maximum(np.asarray(stds, dtype=float), 1e-6)

    z = (observed - means) / stds
    nll = 0.5 * z**2
    total = float(np.sum(nll))

    return total, z, nll