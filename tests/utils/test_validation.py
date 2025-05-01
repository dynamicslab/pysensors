"""
Tests for validation and computation of reconstruction scores and errors.
"""

import numpy as np
import pytest
from scipy.sparse import lil_matrix

from pysensors.utils._validation import determinant, relative_reconstruction_error


class TestDeterminant:
    """Tests for the determinant function."""

    def test_square_case(self):
        """Test when p == r (number of sensors equals number of modes)."""
        top_sensors = np.array([0, 3, 5])
        n_features = 10
        basis_matrix = np.random.rand(10, 3)
        p = len(top_sensors)
        n, r = np.shape(basis_matrix)
        c = lil_matrix((p, n), dtype=np.int8)
        for i in range(p):
            c[i, top_sensors[i]] = 1
        phi = basis_matrix
        theta = c @ phi
        expected = abs(np.linalg.det(theta))
        result = determinant(top_sensors, n_features, basis_matrix)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_overdetermined_case(self):
        """Test when p > r (more sensors than modes)."""
        top_sensors = np.array([0, 3, 5, 7, 9])
        n_features = 10
        basis_matrix = np.random.rand(10, 3)
        p = len(top_sensors)
        n, r = np.shape(basis_matrix)
        c = lil_matrix((p, n), dtype=np.int8)
        for i in range(p):
            c[i, top_sensors[i]] = 1
        phi = basis_matrix
        theta = c @ phi
        expected = abs(np.linalg.det(theta.T @ theta))
        result = determinant(top_sensors, n_features, basis_matrix)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_underdetermined_case(self):
        """Test when p < r (fewer sensors than modes) - should raise an error."""
        top_sensors = np.array([0, 3])
        n_features = 10
        basis_matrix = np.random.rand(10, 3)
        with pytest.raises(UnboundLocalError):
            determinant(top_sensors, n_features, basis_matrix)

    def test_edge_case_single_sensor(self):
        """Test with a single sensor."""
        top_sensors = np.array([5])
        n_features = 10
        basis_matrix = np.random.rand(10, 1)
        p = len(top_sensors)
        n, r = np.shape(basis_matrix)
        c = lil_matrix((p, n), dtype=np.int8)
        for i in range(p):
            c[i, top_sensors[i]] = 1
        phi = basis_matrix
        theta = c @ phi
        expected = abs(np.linalg.det(theta))
        result = determinant(top_sensors, n_features, basis_matrix)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"


class TestRelativeReconstructionError:
    """Tests for the relative_reconstruction_error function."""

    def test_identical_arrays(self):
        """Test with identical data and prediction arrays."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 0.0
        result = relative_reconstruction_error(data, prediction)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_different_arrays(self):
        """Test with different data and prediction arrays."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        expected = (np.linalg.norm(data - prediction) / np.linalg.norm(data)) * 100
        result = relative_reconstruction_error(data, prediction)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_multidimensional_arrays(self):
        """Test with multidimensional arrays."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([[1.2, 2.2], [3.2, 4.2]])
        expected = (np.linalg.norm(data - prediction) / np.linalg.norm(data)) * 100
        result = relative_reconstruction_error(data, prediction)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_edge_case_zero_data(self):
        """Test behavior when data norm is close to zero."""
        data = np.array([0.0, 0.0, 0.0])
        prediction = np.array([0.1, 0.1, 0.1])
        with pytest.warns(RuntimeWarning):
            relative_reconstruction_error(data, prediction)

    def test_large_values(self):
        """Test with very large values."""
        data = np.array([1e10, 2e10, 3e10])
        prediction = np.array([1.01e10, 2.01e10, 3.01e10])
        expected = (np.linalg.norm(data - prediction) / np.linalg.norm(data)) * 100
        result = relative_reconstruction_error(data, prediction)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    top_sensors = np.array([0, 3, 5])
    n_features = 10
    basis_matrix = np.random.rand(10, 3)
    return top_sensors, n_features, basis_matrix


def test_input_types(sample_data):
    """Test that functions handle different input types correctly."""
    top_sensors, n_features, basis_matrix = sample_data
    top_sensors_list = top_sensors.tolist()
    result = determinant(top_sensors_list, n_features, basis_matrix)
    assert isinstance(result, float)
