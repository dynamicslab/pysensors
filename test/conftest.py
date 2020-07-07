"""
Shared pytest fixtures for unit tests.
"""
import numpy as np
import pytest


@pytest.fixture
def data_vandermonde():
    r = 11
    n = 200
    x = np.linspace(0, 1, n + 1)
    vde = np.zeros((n + 1, r))
    vde[:, 0] = np.ones(n + 1)

    for i in range(r - 1):
        vde[:, i + 1] = vde[:, i] * x

    # PySensor objects expect rows to correspond to examples,
    # columns to positions
    return vde.T


@pytest.fixture
def data_random():
    n_examples = 30
    n_features = 20

    return np.random.randn(n_examples, n_features)
