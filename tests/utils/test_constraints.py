# TODO: include some unit tests once there are more functions
# in this submodule
import numpy as np
import pandas as pd
import pytest

# from pysensors.utils._constraints import constraints_eval
# from pysensors.utils._constraints import check_constraints
from pysensors.utils._constraints import (
    get_constrained_sensors_indices,
    get_constrained_sensors_indices_dataframe,
    get_coordinates_from_indices,
    get_indices_from_coordinates,
    load_functional_constraints,
    order_constrained_sensors,
)

# from pysensors.utils._constraints import BaseConstraint


# Testing get_constrained_sensors_indices
def test_get_constrained_sensors_indices_empty_array():
    all_sensors = np.array([])
    x_min, x_max, y_min, y_max, nx, ny = 0, 10, 0, 10, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_non_integer_values():
    all_sensors = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    x_min, x_max, y_min, y_max, nx, ny = 2, 4, 3, 5, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_no_constrained_sensors():
    all_sensors = np.array([[1, 2], [3, 4], [5, 6]])
    x_min, x_max, y_min, y_max, nx, ny = 6, 8, 9, 11, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_single_constrained_sensor():
    x_min, x_max, y_min, y_max, nx, ny = 3, 4, 3, 4, 10, 10
    all_sensors = np.array([i + 101 for i in range(nx * ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_multiple_constrained_sensors():
    all_sensors = np.array(
        [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5], [9.5, 10.5]]
    )
    x_min, x_max, y_min, y_max, nx, ny = 3, 7, 3, 7, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_valid_input_parameters():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 2, 8, 2, 8
    all_sensors = np.array([i for i in range(nx * ny)])
    result = get_constrained_sensors_indices(
        x_min, x_max, y_min, y_max, nx, ny, all_sensors
    )
    assert len(result) == (x_max - x_min + 1) * (y_max - y_min + 1)


def test_one_constrained_sensor():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 8, 9, 8, 9
    all_sensors = np.array([i for i in range(nx * ny)])
    result = get_constrained_sensors_indices(
        x_min, x_max, y_min, y_max, nx, ny, all_sensors
    )
    assert len(result) == 4


def test_invalid_nx_not_integer():
    nx, ny, x_min, x_max, y_min, y_max = "ten", 10, 2, 8, 2, 8
    all_sensors = np.array([i for i in range(ny**2)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_invalid_ny_not_integer():
    nx, ny, x_min, x_max, y_min, y_max = 10, "ten", 2, 8, 2, 8
    all_sensors = np.array([i for i in range(nx**2)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_invalid_x_min_greater_than_x_max():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 8, 2, 2, 8
    all_sensors = np.array([i for i in range(nx * ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


def test_invalid_y_min_greater_than_y_max():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 2, 8, 8, 2
    all_sensors = np.array([i for i in range(nx * ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(  # noqa:F841
            x_min, x_max, y_min, y_max, nx, ny, all_sensors
        )


# Testing get_constrained_sensors_indices_dataframe
def test_get_constrained_sensors_indices_dataframe_does_not_modify_input_dataframe():
    seed = 8051977  # noqa:F841
    # Create a test dataframe
    test_dataframe = pd.DataFrame(
        {
            "x": np.random.randint(0, 100, size=100),
            "y": np.random.randint(0, 100, size=100),
            "Field": np.random.randint(0, 100, size=100),
        }
    )
    df = test_dataframe.copy()
    # Define test parameters
    x_min, x_max, y_min, y_max = 50, 75, 25, 50

    # Call the function
    idx_constrained = get_constrained_sensors_indices_dataframe(  # noqa:F841
        x_min, x_max, y_min, y_max, test_dataframe, X_axis="x", Y_axis="y"
    )

    # Assert that the input dataframe is not modified
    assert test_dataframe.equals(df)


def test_get_constrained_sensors_indices_dataframe():
    """
    Test that the function handles normal constraint.
    """
    x_min, x_max, y_min, y_max = 10, 20, 10, 20
    data = pd.DataFrame({"X_axis": [10, 20, 8, 15, 25], "Y_axis": [10, 32, 20, 18, 12]})
    expected_output = [0, 3]
    assert (
        get_constrained_sensors_indices_dataframe(
            x_min, x_max, y_min, y_max, data, X_axis="X_axis", Y_axis="Y_axis"
        )
        == expected_output
    )


def test_get_constrained_sensors_indices_dataframe_outside_dataframe_range():
    """
    Test that the function handles constraint outside the dataframe range.
    """
    x_min, x_max, y_min, y_max = 0, 5, 0, 5
    data = pd.DataFrame({"X_axis": [10, 15, 20, 25], "Y_axis": [10, 15, 20, 25]})
    expected_output = []
    assert (
        get_constrained_sensors_indices_dataframe(
            x_min, x_max, y_min, y_max, data, X_axis="X_axis", Y_axis="Y_axis"
        )
        == expected_output
    )


def test_get_constrained_sensors_indices_dataframe_overlapping_dataframe_range():
    """
    Test that the function handles constraint overlapping the dataframe range.
    """
    x_min, x_max, y_min, y_max = 15, 25, 15, 25
    data = pd.DataFrame({"X_axis": [10, 15, 20, 25], "Y_axis": [10, 15, 20, 25]})
    expected_output = [1, 2]
    assert (
        get_constrained_sensors_indices_dataframe(
            x_min, x_max, y_min, y_max, data, X_axis="X_axis", Y_axis="Y_axis"
        )
        == expected_output
    )


def test_get_constrained_sensors_indices_dataframe_empty_dataframe():
    """
    Test that the function handles empty dataframe.
    """
    empty_dataframe = pd.DataFrame({"X_axis": [], "Y_axis": []})
    expected_output = []
    assert (
        get_constrained_sensors_indices_dataframe(
            10, 20, 10, 20, empty_dataframe, X_axis="X_axis", Y_axis="Y_axis"
        )
        == expected_output
    )


def test_get_constrained_sensors_indices_dataframe_dataframe_with_missing_values():
    """
    Test that the function handles dataframe with missing values.
    """
    dataframe_with_missing_values = pd.DataFrame(
        {"X_axis": [10, 15, np.nan, 12], "Y_axis": [10, 15, 20, 15]}
    )
    expected_output = [0, 1, 2]
    assert (
        get_constrained_sensors_indices_dataframe(
            10,
            20,
            10,
            20,
            dataframe_with_missing_values,
            X_axis="X_axis",
            Y_axis="Y_axis",
        )
        == expected_output
    )


# Testing order_constrained_sensors
def test_order_constrained_sensors():
    # Define constrained sensor locations
    idx_constrained_list = [1, 2, 3, 4, 5]

    # Define ranks of constrained sensor locations
    ranks_list = [4, 2, 1, 3, 5]
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )

    # Check the results
    assert np.array_equal(
        sortedConstraints, np.array([3, 2, 4, 1, 5])
    ), "Ordering test failed for sortedConstraints"
    assert np.array_equal(
        ranks, np.array([1, 2, 3, 4, 5])
    ), "Ordering test failed for ranks"


def test_order_constrained_sensors_with_reversed_ranks():
    # Define constrained sensor locations
    idx_constrained_list = np.array([1, 2, 3, 4, 5])
    # Define ranks of constrained sensor locations
    ranks_list = np.array([5, 4, 3, 2, 1])

    # Call the function
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )

    # Check the results
    assert np.array_equal(
        sortedConstraints, np.array([5, 4, 3, 2, 1])
    ), "Ordering test failed for sortedConstraints with reversed ranks"
    assert np.array_equal(
        ranks, np.array([1, 2, 3, 4, 5])
    ), "Ordering test failed for ranks with reversed ranks"


def test_order_constrained_sensors_with_empty_ranks_list():
    # Define constrained sensor locations
    idx_constrained_list = np.array([1, 2, 3, 4, 5])

    # Define empty ranks of constrained sensor locations
    ranks_list = []

    # Call the function
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )

    # Check the results
    assert len(sortedConstraints) == 0, "Empty ranks test failed for sortedConstraints"
    assert len(ranks) == 0, "Empty ranks test failed for ranks"


def test_order_constrained_sensors_with_negative_ranks_list():
    # Define constrained sensor locations
    idx_constrained_list = np.array([1, 2, 3, 4, 5])

    # Define empty ranks of constrained sensor locations
    ranks_list = [-3, -2, -5, -1, 0]

    # Call the function
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )

    # Check the results
    assert np.array_equal(
        sortedConstraints, np.array([3, 1, 2, 4, 5])
    ), "Ordering test failed for sortedConstraints with reversed ranks"
    assert np.array_equal(
        ranks, np.array([-5, -3, -2, -1, 0])
    ), "Ordering test failed for ranks with reversed ranks"


# Testing get_coordinates_from_indices
def test_get_coordinates_from_indices_with_numpy_array_info():
    # Define sensor IDs
    idx = np.array([1, 2, 3, 4])

    # Define information
    info = pd.DataFrame({"X_axis": [1, 2, 3, 4, 5], "Y_axis": [10, 20, 30, 40, 50]})
    # Call the function
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis"
    )

    # Check the results
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 2, "Coordinates are not a 2-tuple"


def test_get_coordinates_from_indices_with_pandas_dataframe_info():
    # Define sensor IDs
    idx = np.array([1, 2, 3, 4])

    # Define information
    info = pd.DataFrame({"X_axis": [1, 2, 3, 4, 5], "Y_axis": [10, 20, 30, 40, 50]})

    # Call the function
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis"
    )

    # Check the results
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 2, "Coordinates are not a 2-tuple"
    assert isinstance(coordinates[0], np.ndarray), "X-coordinate is not a numpy array"
    assert isinstance(coordinates[1], np.ndarray), "Y-coordinate is not a numpy array"


def test_get_coordinates_from_indices_with_z_axis():
    # Define sensor IDs
    idx = np.array([1, 2, 3, 4])
    # Define information
    info = pd.DataFrame(
        {
            "X_axis": [1, 2, 3, 4, 5],
            "Y_axis": [10, 20, 30, 40, 50],
            "Z_axis": [1, 2, 3, 4, 5],
        }
    )

    # Call the function
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis", Z_axis="Z_axis"
    )

    # Check the results
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 3, "Coordinates are not a 3-tuple"
    assert isinstance(coordinates[0], np.ndarray), "X-coordinate is not a numpy array"
    assert isinstance(coordinates[1], np.ndarray), "Y-coordinate is not a numpy array"
    assert isinstance(coordinates[2], np.ndarray), "Z-coordinate is not a numpy array"


def test_get_indices_from_coordinates_with_different_shape():
    # Define coordinates
    coordinates = np.array([[3, 6, 6], [4, 5, 1]])

    # Define shape
    shape = (7, 6)

    # Call the function
    indices = get_indices_from_coordinates(coordinates, shape)

    # Check the results
    assert indices.shape == (3,), "Indices shape is not (3,)"
    assert np.array_equal(
        indices, np.array([31, 41, 13])
    ), "Indices test failed with different shape"


# Test load_functional_constraints
def test_load_functional_constraints_loads_valid_python_file():
    """
    Test that the function loads a valid Python file and returns a callable function.
    """
    import os.path

    test_file = "user_function.py"
    abspath = os.path.dirname(os.path.realpath(__file__))
    # abspath = os.getcwd()  # Get absolule path of current work directory
    final_path = abspath + "/" + test_file
    with open(final_path, "w") as f:
        f.write(
            """
def user_function():
    return 1"""
        )
    func = load_functional_constraints(test_file)
    assert func.__name__ == "user_function"
    assert func() == 1


if __name__ == "__main__":
    pytest.main([__file__])
