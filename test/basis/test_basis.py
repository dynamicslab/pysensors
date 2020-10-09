"""Unit tests for basis classes"""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pysensors.basis import Identity
from pysensors.basis import RandomProjection
from pysensors.basis import SVD


@pytest.mark.parametrize("basis", [Identity(), SVD(), RandomProjection()])
def test_not_fitted(basis):
    with pytest.raises(NotFittedError):
        basis.matrix_representation()


def test_identity_matrix_representation(data_random):
    matrix = data_random

    basis = Identity()
    basis.fit(matrix)

    np.testing.assert_allclose(matrix.T, basis.matrix_representation())


@pytest.mark.parametrize("basis", [SVD, RandomProjection])
def test_matrix_representation(basis, data_random):
    data = data_random
    n_features = data.shape[1]
    n_components = 5

    b = basis(n_basis_modes=n_components)
    b.fit(data)
    matrix_representation = b.matrix_representation()

    assert matrix_representation.shape[0] == n_features
    assert matrix_representation.shape[1] == n_components


def test_random_projection_random_state(data_vandermonde):
    data = data_vandermonde

    basis1 = RandomProjection(n_basis_modes=5, random_state=1)
    m1 = basis1.fit(data).matrix_representation()

    basis2 = RandomProjection(n_basis_modes=5, random_state=2)
    m2 = basis2.fit(data).matrix_representation()

    assert not np.allclose(m1, m2)


@pytest.mark.parametrize("basis", [Identity, SVD, RandomProjection])
def test_n_basis_modes(basis, data_random):
    with pytest.raises(ValueError):
        b = basis(n_basis_modes=0)
    with pytest.raises(ValueError):
        b = basis(n_basis_modes=1.2)
    with pytest.raises(ValueError):
        b = basis(n_basis_modes="1")

    data = data_random
    n_basis_modes = 5
    b = basis(n_basis_modes=n_basis_modes)
    b.fit(data)

    assert b.matrix_representation().shape[1] == n_basis_modes


@pytest.mark.parametrize("basis", [Identity, SVD])
def test_extra_basis_modes(basis, data_random):
    data = data_random
    n_basis_modes = data.shape[0] + 1
    b = basis(n_basis_modes=n_basis_modes)
    # Can't have more basis modes than the number of training examples
    with pytest.raises(ValueError):
        b.fit(data)


@pytest.mark.parametrize("basis", [SVD(), RandomProjection()])
def test_matrix_inverse_shape(basis, data_random):
    data = data_random
    n_features = data.shape[1]
    n_basis_modes = 5

    basis.fit(data)
    inverse = basis.matrix_inverse(n_basis_modes=n_basis_modes)

    assert inverse.shape == (n_basis_modes, n_features)
