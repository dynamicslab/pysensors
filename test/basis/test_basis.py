"""Unit tests for basis classes"""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pysensors.basis import Identity


def test_fitted(data_vandermonde):
    basis = Identity()

    with pytest.raises(NotFittedError):
        basis.matrix_representation()


def test_matrix_representation():
    matrix = np.random.randn(10, 15)

    basis = Identity()
    basis.fit(matrix)

    np.testing.assert_allclose(matrix.T, basis.matrix_representation())
