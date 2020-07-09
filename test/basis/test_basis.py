"""Unit tests for basis classes"""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pysensors.basis import Identity
from pysensors.basis import POD


@pytest.mark.parametrize("basis", [Identity(), POD()])
def test_not_fitted(basis):
    with pytest.raises(NotFittedError):
        basis.matrix_representation()


def test_identity_matrix_representation(data_random):
    matrix = data_random

    basis = Identity()
    basis.fit(matrix)

    np.testing.assert_allclose(matrix.T, basis.matrix_representation())


def test_pod_matrix_representation(data_random):
    data = data_random
    n_features = data.shape[1]
    n_components = 5

    basis = POD(n_basis_modes=n_components)
    basis.fit(data)
    matrix_representation = basis.matrix_representation()

    assert matrix_representation.shape[0] == n_features
    assert matrix_representation.shape[1] == n_components
