"""Unit tests for basis classes"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pysensors.basis import SVD, Custom, Identity, RandomProjection, _base


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

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


@pytest.fixture
def sample_basis():
    """Create a sample basis matrix for testing."""
    return np.eye(5)


def test_valid_n_basis_modes(sample_basis):
    """Test initialization with valid n_basis_modes."""
    custom = Custom(U=sample_basis, n_basis_modes=3)
    assert custom._n_basis_modes == 3
    np.testing.assert_array_equal(custom.custom_basis_, sample_basis)

    custom = Custom(U=sample_basis, n_basis_modes=1)
    assert custom._n_basis_modes == 1


@pytest.mark.parametrize("value", [3.5, "3", [3], (3,), None])
def test_n_basis_modes_not_integer(sample_basis, value):
    """Test initialization with non-integer n_basis_modes."""
    with pytest.raises(ValueError) as excinfo:
        Custom(U=sample_basis, n_basis_modes=value)
    assert "n_basis_modes must be a positive integer" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, -1, -10])
def test_n_basis_modes_not_positive(sample_basis, value):
    """Test initialization with non-positive n_basis_modes."""
    with pytest.raises(ValueError) as excinfo:
        Custom(U=sample_basis, n_basis_modes=value)
    assert "n_basis_modes must be a positive integer" in str(excinfo.value)


def test_with_keyword_arguments(sample_basis):
    """Test initialization with additional keyword arguments."""
    custom = Custom(
        U=sample_basis, n_basis_modes=3, extra_param=True, another_param="value"
    )
    assert custom._n_basis_modes == 3


@pytest.fixture
def custom_instance(sample_basis):
    """Create an initialized Custom instance for testing."""
    return Custom(U=sample_basis, n_basis_modes=3)


def test_fit_method(custom_instance, sample_basis):
    """Test that fit correctly sets basis_matrix_ and returns self."""
    X = np.ones(sample_basis.shape)
    result = custom_instance.fit(X)
    expected_basis_matrix = sample_basis[:, :3]
    np.testing.assert_array_equal(custom_instance.basis_matrix_, expected_basis_matrix)
    assert result is custom_instance


def test_matrix_inverse_default(custom_instance):
    """Test matrix_inverse with default n_basis_modes."""
    X = np.random.random((10, 10))
    custom_instance.fit(X)
    result = custom_instance.matrix_inverse()
    expected_result = custom_instance.basis_matrix_.T
    np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize("n_modes", [1, 2])
def test_matrix_inverse_with_n_basis_modes(custom_instance, n_modes):
    """Test matrix_inverse with specified n_basis_modes."""
    X = np.random.random((10, 10))
    custom_instance.fit(X)
    result = custom_instance.matrix_inverse(n_basis_modes=n_modes)
    expected_result = custom_instance.basis_matrix_[:, :n_modes].T
    np.testing.assert_array_equal(result, expected_result)
    assert result.shape == (n_modes, 5)


def test_n_basis_modes_getter(custom_instance):
    """Test n_basis_modes property getter."""
    assert custom_instance.n_basis_modes == 3
    custom_instance._n_basis_modes = 4
    assert custom_instance.n_basis_modes == 4


def test_n_basis_modes_setter(custom_instance):
    """Test n_basis_modes property setter."""
    custom_instance.n_basis_modes = 2
    assert custom_instance._n_basis_modes == 2
    assert custom_instance.n_components == 2

    custom_instance.n_basis_modes = 4
    assert custom_instance._n_basis_modes == 4
    assert custom_instance.n_components == 4


def test_matrix_inverse_calls_validate_input(custom_instance, monkeypatch):
    """Test that matrix_inverse calls _validate_input."""
    X = np.random.random((10, 10))
    custom_instance.fit(X)
    validation_called = False
    test_value = None

    def mock_validate_input(self, value):
        nonlocal validation_called, test_value
        validation_called = True
        test_value = value
        return 2

    monkeypatch.setattr(Custom, "_validate_input", mock_validate_input)
    custom_instance.matrix_inverse(n_basis_modes=3)
    assert validation_called
    assert test_value == 3


def test_invertible_basis_abstract_method():
    class TestBasis(_base.InvertibleBasis):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class TestBasis"):
        TestBasis()

    class ProperTestBasis(_base.InvertibleBasis):
        def matrix_inverse(self, n_basis_modes=None, **kwargs):
            return None

    with pytest.raises(
        NotImplementedError, match="This method has not been implemented"
    ):
        _base.InvertibleBasis.matrix_inverse(None)


def test_validate_input_too_many_modes_error():
    basis = SVD()
    n_available_modes = 5
    basis.basis_matrix_ = np.random.rand(10, n_available_modes)
    basis.n_basis_modes = n_available_modes
    too_many_modes = 8

    expected_error = (
        f"Requested number of modes {too_many_modes} exceeds"
        f" number available: {n_available_modes}"
    )

    with pytest.raises(ValueError, match=expected_error):
        basis._validate_input(n_basis_modes=too_many_modes)


def test_matrix_representation_copy():
    basis = SVD()
    n_features = 10
    n_basis_modes = 5
    basis.basis_matrix_ = np.random.rand(n_features, n_basis_modes)
    result_copy = basis.matrix_representation(n_basis_modes=3, copy=True)
    np.testing.assert_array_equal(result_copy, basis.basis_matrix_[:, :3])
    original_value = basis.basis_matrix_[0, 0]
    result_copy[0, 0] = 999
    assert basis.basis_matrix_[0, 0] == original_value
    result_view = basis.matrix_representation(n_basis_modes=3, copy=False)
    np.testing.assert_array_equal(result_view, basis.basis_matrix_[:, :3])
    result_view[0, 0] = 777
    assert basis.basis_matrix_[0, 0] == 777
