import numpy as np
import pytest

from pysensors.utils import validate_input  # Adjust import path as needed


def test_validate_input_value_errors():
    """Test that validate_input raises appropriate ValueErrors."""
    not_arrays = [[1, 2, 3], (1, 2, 3), {1, 2, 3}, {"a": 1, "b": 2}, "123", 123, None]
    for not_array in not_arrays:
        with pytest.raises(ValueError, match="x must be a numpy array"):
            validate_input(not_array)
    x_1d = np.array([1, 2, 3, 4])
    wrong_sensors_1d = [np.array([0, 1]), np.array([0, 1, 2, 3, 4]), np.array([])]
    for sensors in wrong_sensors_1d:
        with pytest.raises(ValueError, match="x has the wrong number of features"):
            validate_input(x_1d, sensors)
    x_2d = np.array([[1, 2, 3], [4, 5, 6]])
    wrong_sensors_2d = [np.array([0, 1]), np.array([0, 1, 2, 3]), np.array([])]
    for sensors in wrong_sensors_2d:
        with pytest.raises(ValueError, match="x has the wrong number of features"):
            validate_input(x_2d, sensors)


def test_validate_input_valid_cases():
    """Test that validate_input works correctly with valid inputs."""
    x_1d = np.array([1, 2, 3, 4])
    sensors_1d = np.array([0, 1, 2, 3])
    result = validate_input(x_1d, sensors_1d)
    assert np.array_equal(result, x_1d)
    x_2d = np.array([[1, 2, 3], [4, 5, 6]])
    sensors_2d = np.array([0, 1, 2])
    result = validate_input(x_2d, sensors_2d)
    assert np.array_equal(result, x_2d)
    result = validate_input(x_1d, None)
    assert np.array_equal(result, x_1d)

    result = validate_input(x_2d, None)
    assert np.array_equal(result, x_2d)
