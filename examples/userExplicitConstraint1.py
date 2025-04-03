import numpy as np


def userExplicitConstraint1(x, y, **kwargs):
    """ """
    assert len(x) == len(y)
    g = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        g[i] = ((x[i] - 30) ** 2 + (y[i] - 40) ** 2) - 5**2
    return g
