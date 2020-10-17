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
    vde = np.vander(x, r, increasing=True)

    # PySensor objects expect rows to correspond to examples,
    # columns to positions
    return vde.T


# Same as above, but also returns test data
@pytest.fixture
def data_vandermonde_testing():
    r = 11
    n = 200
    x = np.linspace(0, 1, n + 1)
    vde = np.zeros((n + 1, r))
    vde[:, 0] = np.ones(n + 1)

    for i in range(r - 1):
        vde[:, i + 1] = vde[:, i] * x

    v = np.zeros(r)
    v[[1, 3, 5]] = 1
    x_test = np.dot(vde, v)

    # PySensor objects expect rows to correspond to examples,
    # columns to positions
    return vde.T, x_test


@pytest.fixture
def data_random():
    n_examples = 30
    n_features = 20

    return np.random.randn(n_examples, n_features)


@pytest.fixture
def data_random_square():
    n_examples = 30
    n_features = 30

    return np.random.randn(n_examples, n_features)
