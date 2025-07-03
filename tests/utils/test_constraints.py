import os.path
from unittest.mock import ANY, MagicMock, patch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pysensors.utils._constraints import (
    BaseConstraint,
    Circle,
    Cylinder,
    Ellipse,
    Line,
    Parabola,
    Polygon,
    UserDefinedConstraints,
    get_constrained_sensors_indices,
    get_constrained_sensors_indices_dataframe,
    get_constrained_sensors_indices_distance,
    get_constrained_sensors_indices_distance_df,
    get_coordinates_from_indices,
    get_indices_from_coordinates,
    load_functional_constraints,
    order_constrained_sensors,
)


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


def test_get_constrained_sensors_indices_dataframe_does_not_modify_input_dataframe():
    seed = 8051977  # noqa:F841
    test_dataframe = pd.DataFrame(
        {
            "x": np.random.randint(0, 100, size=100),
            "y": np.random.randint(0, 100, size=100),
            "Field": np.random.randint(0, 100, size=100),
        }
    )
    df = test_dataframe.copy()
    x_min, x_max, y_min, y_max = 50, 75, 25, 50
    idx_constrained = get_constrained_sensors_indices_dataframe(  # noqa:F841
        x_min, x_max, y_min, y_max, test_dataframe, X_axis="x", Y_axis="y"
    )
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


def test_order_constrained_sensors():
    idx_constrained_list = [1, 2, 3, 4, 5]
    ranks_list = [4, 2, 1, 3, 5]
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )
    assert np.array_equal(
        sortedConstraints, np.array([3, 2, 4, 1, 5])
    ), "Ordering test failed for sortedConstraints"
    assert np.array_equal(
        ranks, np.array([1, 2, 3, 4, 5])
    ), "Ordering test failed for ranks"


def test_order_constrained_sensors_with_reversed_ranks():
    idx_constrained_list = np.array([1, 2, 3, 4, 5])
    ranks_list = np.array([5, 4, 3, 2, 1])
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )
    assert np.array_equal(
        sortedConstraints, np.array([5, 4, 3, 2, 1])
    ), "Ordering test failed for sortedConstraints with reversed ranks"
    assert np.array_equal(
        ranks, np.array([1, 2, 3, 4, 5])
    ), "Ordering test failed for ranks with reversed ranks"


def test_order_constrained_sensors_with_empty_ranks_list():
    idx_constrained_list = np.array([1, 2, 3, 4, 5])
    ranks_list = []
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )
    assert len(sortedConstraints) == 0, "Empty ranks test failed for sortedConstraints"
    assert len(ranks) == 0, "Empty ranks test failed for ranks"


def test_order_constrained_sensors_with_negative_ranks_list():
    idx_constrained_list = np.array([1, 2, 3, 4, 5])
    ranks_list = [-3, -2, -5, -1, 0]
    sortedConstraints, ranks = order_constrained_sensors(
        idx_constrained_list, ranks_list
    )
    assert np.array_equal(
        sortedConstraints, np.array([3, 1, 2, 4, 5])
    ), "Ordering test failed for sortedConstraints with reversed ranks"
    assert np.array_equal(
        ranks, np.array([-5, -3, -2, -1, 0])
    ), "Ordering test failed for ranks with reversed ranks"


def test_get_coordinates_from_indices_with_numpy_array_info():
    idx = np.array([1, 2, 3, 4])
    info = pd.DataFrame({"X_axis": [1, 2, 3, 4, 5], "Y_axis": [10, 20, 30, 40, 50]})
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis"
    )
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 2, "Coordinates are not a 2-tuple"


def test_get_coordinates_from_indices_with_pandas_dataframe_info():
    idx = np.array([1, 2, 3, 4])
    info = pd.DataFrame({"X_axis": [1, 2, 3, 4, 5], "Y_axis": [10, 20, 30, 40, 50]})
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis"
    )
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 2, "Coordinates are not a 2-tuple"
    assert isinstance(coordinates[0], np.ndarray), "X-coordinate is not a numpy array"
    assert isinstance(coordinates[1], np.ndarray), "Y-coordinate is not a numpy array"


def test_get_coordinates_from_indices_with_z_axis():
    idx = np.array([1, 2, 3, 4])
    info = pd.DataFrame(
        {
            "X_axis": [1, 2, 3, 4, 5],
            "Y_axis": [10, 20, 30, 40, 50],
            "Z_axis": [1, 2, 3, 4, 5],
        }
    )
    coordinates = get_coordinates_from_indices(
        idx, info, X_axis="X_axis", Y_axis="Y_axis", Z_axis="Z_axis"
    )
    assert isinstance(coordinates, tuple), "Coordinates are not a tuple"
    assert len(coordinates) == 3, "Coordinates are not a 3-tuple"
    assert isinstance(coordinates[0], np.ndarray), "X-coordinate is not a numpy array"
    assert isinstance(coordinates[1], np.ndarray), "Y-coordinate is not a numpy array"
    assert isinstance(coordinates[2], np.ndarray), "Z-coordinate is not a numpy array"


def test_get_indices_from_coordinates_with_different_shape():
    coordinates = np.array([[3, 6, 6], [4, 5, 1]])
    shape = (7, 6)
    indices = get_indices_from_coordinates(coordinates, shape)
    assert indices.shape == (3,), "Indices shape is not (3,)"
    assert np.array_equal(
        indices, np.array([31, 41, 13])
    ), "Indices test failed with different shape"


def test_get_constrained_sensors_indices_last_block():
    """Test specifically the last if-else block of get_constrained_sensors_indices."""
    nx, ny = 10, 10
    all_sensors = np.array([0, 1, 20, 30])
    x_min, x_max = 5, 8
    y_min, y_max = 5, 8

    result_empty = get_constrained_sensors_indices(
        x_min, x_max, y_min, y_max, nx, ny, all_sensors
    )
    assert isinstance(result_empty, list)
    assert len(result_empty) == 0
    nx, ny = 10, 10
    all_sensors = np.array([65])
    coords = np.unravel_index([65], (nx, ny))
    x, y = coords[0][0], coords[1][0]  #

    x_min, x_max = x - 1, x + 1
    y_min, y_max = y - 1, y + 1

    result = get_constrained_sensors_indices(
        x_min, x_max, y_min, y_max, nx, ny, all_sensors
    )
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([56]))


def test_get_constrained_sensors_indices_dataframe_exceptions():
    """Test that the function raises exceptions when required kwargs are missing."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
    with pytest.raises(Exception) as excinfo:
        get_constrained_sensors_indices_dataframe(0, 10, 0, 10, df, Y_axis="y")
    assert "Must provide X_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )
    with pytest.raises(Exception) as excinfo:
        get_constrained_sensors_indices_dataframe(0, 10, 0, 10, df, X_axis="x")
    assert "Must provide Y_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )
    try:
        result = get_constrained_sensors_indices_dataframe(
            0, 10, 0, 10, df, X_axis="x", Y_axis="y"
        )
        assert isinstance(result, list)
    except Exception as e:
        pytest.fail(f"Function raised an exception unexpectedly: {e}")


def test_get_coordinates_from_indices_exceptions():
    """Test all possible exceptions raised by get_coordinates_from_indices."""
    df = pd.DataFrame(
        {
            "x_coord": [1, 2, 3, 4, 5],
            "y_coord": [5, 4, 3, 2, 1],
            "z_coord": [10, 20, 30, 40, 50],
        }
    )
    np_data = np.random.rand(4, 16)
    with pytest.raises(Exception) as excinfo:
        get_coordinates_from_indices([0, 10], df, X_axis="x_coord", Y_axis="y_coord")
    assert "Sensor ID must be within dataframe entries" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo:
        get_coordinates_from_indices([0, 1], df, Y_axis="y_coord")
    assert "Must provide X_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )
    with pytest.raises(Exception) as excinfo:
        get_coordinates_from_indices([0, 1], df, X_axis="x_coord")
    assert "Must provide Y_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )
    try:
        result = get_coordinates_from_indices(
            [0, 1], df, X_axis="x_coord", Y_axis="y_coord"
        )
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
    except Exception as e:
        pytest.fail(
            f"Function raised an unexpected exception with valid DataFrame input: {e}"
        )

    try:
        result = get_coordinates_from_indices(
            [0, 1], df, X_axis="x_coord", Y_axis="y_coord", Z_axis="z_coord"
        )
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)
    except Exception as e:
        pytest.fail(f"Function raised an unexpected exception with Z_axis: {e}")

    try:
        result = get_coordinates_from_indices(5, np_data)
        assert len(result) == 2
    except Exception as e:
        pytest.fail(f"Function raised an unexpected exception with numpy array: {e}")


def test_load_functional_constraints_loads_valid_python_file():
    """
    Test that the function loads a valid Python file and returns a callable function.
    """
    test_file = "test_user_function.py"
    abspath = os.path.dirname(os.path.realpath(__file__))
    final_path = abspath + "/" + test_file
    with open(final_path, "w") as f:
        f.write(
            """
def test_user_function():
    return 1"""
        )
    func = load_functional_constraints(test_file)
    assert func.__name__ == "test_user_function"
    assert func() == 1


def test_get_constrained_sensors_indices_distance_empty_piv():
    """Test that the function handles empty piv array."""
    piv = np.array([])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 2.0, 5, 5
    with pytest.raises(IndexError):
        result = get_constrained_sensors_indices_distance(  # noqa:F841
            j, piv, r, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_distance_j_zero():
    """Test that the function handles j=0 correctly."""
    piv = np.array([0, 12, 24])
    all_sensors = np.arange(25)
    j, r, nx, ny = 0, 1.5, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 0) ** 2 + (y - 0) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_center_position():
    """Test that the function handles center sensor correctly."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 2.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 2) ** 2 + (y - 2) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_large_radius():
    """Test that the function handles very large radius correctly."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 10.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    assert np.array_equal(np.sort(result), np.sort(all_sensors))


def test_get_constrained_sensors_indices_distance_small_radius():
    """Test that the function handles very small radius correctly."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 0.5, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected = np.array([12])
    assert np.array_equal(result, expected)


def test_get_constrained_sensors_indices_distance_corner_position():
    """Test that the function handles corner sensor correctly."""
    piv = np.array([0])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 1.5, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 0) ** 2 + (y - 0) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_edge_position():
    """Test that the function handles edge sensor correctly."""
    piv = np.array([2])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 1.5, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 0) ** 2 + (y - 2) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_different_grid_size():
    """Test that the function handles different grid dimensions."""
    nx, ny = 3, 4
    all_sensors = np.arange(nx * ny)
    piv = np.array([5])
    j, r = 1, 1.5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    x_ref, y_ref = np.unravel_index(5, (nx, ny))
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - x_ref) ** 2 + (y - y_ref) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_boundary_radius():
    """Test that the function handles radius exactly on distance boundary."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, nx, ny = 1, 5, 5
    r = np.sqrt(2)
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 2) ** 2 + (y - 2) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_single_sensor_grid():
    """Test that the function handles 1x1 grid."""
    nx, ny = 1, 1
    all_sensors = np.array([0])
    piv = np.array([0])
    j, r = 1, 1.0
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected = np.array([0])
    assert np.array_equal(result, expected)


def test_get_constrained_sensors_indices_distance_no_sensors_in_radius():
    """Test that the function handles case where no sensors are within radius."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 0.1, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    assert len(result) <= 1
    if len(result) == 1:
        assert result[0] == 12


def test_get_constrained_sensors_indices_distance_large_j_value():
    """Test that the function raises IndexError when j causes out-of-bounds access."""
    piv = np.array([5, 10, 15])
    all_sensors = np.arange(25)
    j, r, nx, ny = 10, 2.0, 5, 5
    with pytest.raises(IndexError):
        result = get_constrained_sensors_indices_distance(  # noqa:F841
            j, piv, r, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_distance_subset_sensors():
    """Test that the function handles subset of sensors correctly."""
    all_sensors = np.array([0, 5, 10, 15, 20])
    piv = np.array([10])
    j, r, nx, ny = 1, 3.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    assert all(sensor in all_sensors for sensor in result)
    x_ref, y_ref = np.unravel_index(10, (nx, ny))
    for sensor in result:
        x, y = np.unravel_index(sensor, (nx, ny))
        distance_sq = (x - x_ref) ** 2 + (y - y_ref) ** 2
        assert distance_sq < r**2


def test_get_constrained_sensors_indices_distance_invalid_sensor_index():
    """Test that the function handles invalid sensor indices in piv."""
    piv = np.array([100])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 2.0, 5, 5
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices_distance(  # noqa:F841
            j, piv, r, nx, ny, all_sensors
        )


def test_get_constrained_sensors_indices_distance_negative_radius():
    """Test that the function handles negative radius."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, -1.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    assert len(result) >= 1


def test_get_constrained_sensors_indices_distance_zero_radius():
    """Test that the function handles zero radius."""
    piv = np.array([12])
    all_sensors = np.arange(25)
    j, r, nx, ny = 1, 0.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    assert len(result) == 0


def test_get_constrained_sensors_indices_distance_rectangular_grid():
    """Test that the function handles rectangular (non-square) grids."""
    nx, ny = 2, 8
    all_sensors = np.arange(nx * ny)
    piv = np.array([5])
    j, r = 1, 2.0
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected_indices = []
    for idx in all_sensors:
        x, y = np.unravel_index(idx, (nx, ny))
        if (x - 0) ** 2 + (y - 5) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_single_sensor_in_all_sensors():
    """Test that the function handles single sensor in all_sensors array."""
    piv = np.array([12])
    all_sensors = np.array([12])
    j, r, nx, ny = 1, 2.0, 5, 5
    result = get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors)
    expected = np.array([12])
    assert np.array_equal(result, expected)


def test_get_constrained_sensors_indices_distance_df_empty_dataframe():
    """Test that the function handles empty DataFrame."""
    piv = np.array([0])
    df = pd.DataFrame({}, columns=["x", "y"])
    all_sensors = np.array([])
    j, r = 1, 2.0
    with pytest.raises(KeyError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "x", "y"
        )


def test_get_constrained_sensors_indices_distance_df_empty_all_sensors():
    """Test that the function handles empty all_sensors array."""
    piv = np.array([0])
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    all_sensors = np.array([])
    j, r = 1, 2.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert len(result) == 0


def test_get_constrained_sensors_indices_distance_df_empty_piv():
    """Test that the function handles empty piv array."""
    piv = np.array([])
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 2.0
    with pytest.raises(IndexError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "x", "y"
        )


def test_get_constrained_sensors_indices_distance_df_j_zero():
    """Test that the function handles j=0 correctly."""
    piv = np.array([0, 1, 2])
    df = pd.DataFrame({"x": [0, 5, 10], "y": [0, 0, 0]})
    all_sensors = np.array([0, 1, 2])
    j, r = 0, 3.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected_indices = []
    current_x, current_y = 0, 0
    for idx in all_sensors:
        x, y = df.loc[idx, "x"], df.loc[idx, "y"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_basic_functionality():
    """Test basic functionality with simple DataFrame."""
    piv = np.array([1])
    df = pd.DataFrame({"x": [0, 5, 10, 15, 20], "y": [0, 0, 0, 0, 0]})
    all_sensors = np.array([0, 1, 2, 3, 4])
    j, r = 1, 6.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected_indices = []
    current_x, current_y = 5, 0
    for idx in all_sensors:
        x, y = df.loc[idx, "x"], df.loc[idx, "y"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_large_radius():
    """Test that the function handles very large radius correctly."""
    piv = np.array([2])
    df = pd.DataFrame({"x": [0, 5, 10, 15, 20], "y": [0, 5, 10, 15, 20]})
    all_sensors = np.array([0, 1, 2, 3, 4])
    j, r = 1, 100.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert np.array_equal(np.sort(result), np.sort(all_sensors))


def test_get_constrained_sensors_indices_distance_df_small_radius():
    """Test that the function handles very small radius correctly."""
    piv = np.array([2])
    df = pd.DataFrame({"x": [0, 5, 10, 15, 20], "y": [0, 5, 10, 15, 20]})
    all_sensors = np.array([0, 1, 2, 3, 4])
    j, r = 1, 0.5
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected = np.array([2])
    assert np.array_equal(result, expected)


def test_get_constrained_sensors_indices_distance_df_missing_column():
    """Test that the function handles missing column names."""
    piv = np.array([0])
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 2.0
    with pytest.raises(KeyError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "z", "y"
        )


def test_get_constrained_sensors_indices_distance_df_invalid_sensor_index():
    """Test that the function handles invalid sensor indices in piv."""
    piv = np.array([10])
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 2.0
    with pytest.raises(KeyError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "x", "y"
        )


def test_get_constrained_sensors_indices_distance_df_invalid_all_sensors_index():
    """Test that the function handles invalid indices in all_sensors."""
    piv = np.array([0])
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    all_sensors = np.array([0, 1, 10])
    j, r = 1, 2.0
    with pytest.raises(KeyError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "x", "y"
        )


def test_get_constrained_sensors_indices_distance_df_large_j_value():
    """Test that the function raises IndexError when j causes out-of-bounds access."""
    piv = np.array([0, 1, 2])
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    all_sensors = np.array([0, 1, 2, 3, 4])
    j, r = 10, 2.0
    with pytest.raises(IndexError):
        result = get_constrained_sensors_indices_distance_df(  # noqa:F841
            j, piv, r, df, all_sensors, "x", "y"
        )


def test_get_constrained_sensors_indices_distance_df_large_j_value_valid():
    """Test that the function handles j larger than expected but within piv bounds."""
    piv = np.array([0, 2, 4])
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]})
    all_sensors = np.array([0, 1, 2, 3, 4])
    j, r = 3, 2.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected_indices = []
    current_x, current_y = 4, 4
    for idx in all_sensors:
        x, y = df.loc[idx, "x"], df.loc[idx, "y"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_subset_sensors():
    """Test that the function handles subset of sensors correctly."""
    piv = np.array([2])
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 2, 3, 4, 5]})
    all_sensors = np.array([0, 2, 4])
    j, r = 1, 3.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert all(sensor in all_sensors for sensor in result)
    current_x, current_y = df.loc[2, "x"], df.loc[2, "y"]
    for sensor in result:
        x, y = df.loc[sensor, "x"], df.loc[sensor, "y"]
        distance_sq = (x - current_x) ** 2 + (y - current_y) ** 2
        assert distance_sq < r**2


def test_get_constrained_sensors_indices_distance_df_negative_radius():
    """Test that the function handles negative radius."""
    piv = np.array([1])
    df = pd.DataFrame({"x": [0, 5, 10], "y": [0, 0, 0]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, -1.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert len(result) >= 1


def test_get_constrained_sensors_indices_distance_df_zero_radius():
    """Test that the function handles zero radius."""
    piv = np.array([1])
    df = pd.DataFrame({"x": [0, 5, 10], "y": [0, 0, 0]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 0.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert len(result) == 0


def test_get_constrained_sensors_indices_distance_df_different_column_names():
    """Test that the function handles different column names."""
    piv = np.array([1])
    df = pd.DataFrame({"longitude": [0, 5, 10], "latitude": [0, 5, 10]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 8.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "longitude", "latitude"
    )
    expected_indices = []
    current_x, current_y = 5, 5
    for idx in all_sensors:
        x, y = df.loc[idx, "longitude"], df.loc[idx, "latitude"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_float_coordinates():
    """Test that the function handles float coordinates."""
    piv = np.array([1])
    df = pd.DataFrame({"x": [0.5, 2.3, 4.7, 6.1], "y": [1.2, 3.4, 5.6, 7.8]})
    all_sensors = np.array([0, 1, 2, 3])
    j, r = 1, 3.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected_indices = []
    current_x, current_y = 2.3, 3.4
    for idx in all_sensors:
        x, y = df.loc[idx, "x"], df.loc[idx, "y"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_with_nan_values():
    """Test that the function handles NaN values in DataFrame."""
    piv = np.array([0])
    df = pd.DataFrame({"x": [1, np.nan, 3, 4], "y": [1, 2, np.nan, 4]})
    all_sensors = np.array([0, 1, 2, 3])
    j, r = 1, 2.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    for sensor in result:
        x, y = df.loc[sensor, "x"], df.loc[sensor, "y"]
        assert not (pd.isna(x) or pd.isna(y))


def test_get_constrained_sensors_indices_distance_df_single_sensor():
    """Test that the function handles single sensor in all_sensors array."""
    piv = np.array([1])
    df = pd.DataFrame({"x": [0, 5, 10], "y": [0, 5, 10]})
    all_sensors = np.array([1])
    j, r = 1, 2.0
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected = np.array([1])
    assert np.array_equal(result, expected)


def test_get_constrained_sensors_indices_distance_df_boundary_radius():
    """Test that the function handles radius exactly on distance boundary."""
    piv = np.array([0])
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1]})
    all_sensors = np.array([0, 1, 2, 3])
    j, r = 1, np.sqrt(2)
    result = get_constrained_sensors_indices_distance_df(
        j, piv, r, df, all_sensors, "x", "y"
    )
    expected_indices = []
    current_x, current_y = 0, 0
    for idx in all_sensors:
        x, y = df.loc[idx, "x"], df.loc[idx, "y"]
        if (x - current_x) ** 2 + (y - current_y) ** 2 < r**2:
            expected_indices.append(idx)
    expected = np.array(expected_indices)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_get_constrained_sensors_indices_distance_df_does_not_modify_input():
    """Test that the function does not modify input DataFrame or arrays."""
    piv = np.array([0, 1])
    df = pd.DataFrame({"x": [0, 5, 10], "y": [0, 5, 10]})
    all_sensors = np.array([0, 1, 2])
    j, r = 1, 6.0
    piv_copy = piv.copy()
    df_copy = df.copy()
    all_sensors_copy = all_sensors.copy()

    result = get_constrained_sensors_indices_distance_df(  # noqa:F841
        j, piv, r, df, all_sensors, "x", "y"
    )
    assert np.array_equal(piv, piv_copy)
    assert df.equals(df_copy)
    assert np.array_equal(all_sensors, all_sensors_copy)


class TestBaseConstraint:

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "z_coord": [10, 20, 30, 40, 50],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init_with_dataframe(self, sample_dataframe):
        """Test initialization with a dataframe."""
        constraint = BaseConstraint(
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
            Z_axis="z_coord",
        )

        assert constraint.data.equals(sample_dataframe)
        assert constraint.X_axis == "x_coord"
        assert constraint.Y_axis == "y_coord"
        assert constraint.Z_axis == "z_coord"
        assert constraint.Field == "field_val"

    def test_init_exceptions(self, sample_dataframe):
        """Test initialization exceptions."""
        with pytest.raises(Exception) as excinfo:
            BaseConstraint()
        assert "Must provide data as **kwargs" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            BaseConstraint(data=sample_dataframe, Y_axis="y_coord", Field="field_val")
        assert "Must provide X_axis as **kwargs as your data is a dataframe" in str(
            excinfo.value
        )
        with pytest.raises(Exception) as excinfo:
            BaseConstraint(data=sample_dataframe, X_axis="x_coord", Field="field_val")
        assert "Must provide Y_axis as **kwargs as your data is a dataframe" in str(
            excinfo.value
        )
        with pytest.raises(Exception) as excinfo:
            BaseConstraint(data=sample_dataframe, X_axis="x_coord", Y_axis="y_coord")
        assert "Must provide Field as **kwargs as your data is a dataframe" in str(
            excinfo.value
        )

    def test_functional_constraints_with_numpy_array(self, sample_array):
        """Test functional_constraints with numpy array."""
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2]), np.array([3, 4]))

            def test_func(x, y, **kwargs):
                return x > y

            result = BaseConstraint.functional_constraints(
                test_func, np.array([0, 1]), sample_array
            )
            mock_get_coords.assert_called_once_with(ANY, sample_array)
            np.testing.assert_array_equal(result, np.array([False, False]))

    def test_functional_constraints_with_dataframe(self, sample_dataframe):
        """Test functional_constraints with dataframe."""
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2]), np.array([3, 4]))

            def test_func(x, y, **kwargs):
                return x > y

            result = BaseConstraint.functional_constraints(
                test_func,
                np.array([0, 1]),
                sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
                Z_axis="z_coord",
            )
            mock_get_coords.assert_called_once_with(
                ANY,
                ANY,
                X_axis="x_coord",
                Y_axis="y_coord",
                Z_axis="z_coord",
                Field="field_val",
            )
            args, kwargs = mock_get_coords.call_args
            np.testing.assert_array_equal(args[0], np.array([0, 1]))
            assert args[1].equals(sample_dataframe)
            np.testing.assert_array_equal(result, np.array([False, False]))

    def test_functional_constraints_missing_kwargs(self, sample_dataframe):
        """Test exceptions in functional_constraints with missing kwargs."""

        def test_func(x, y, **kwargs):
            return x > y

        with pytest.raises(Exception) as excinfo:
            BaseConstraint.functional_constraints(
                test_func,
                np.array([0, 1]),
                sample_dataframe,
                Y_axis="y_coord",
                Field="field_val",
            )
        assert "Must provide X_axis as **kwargs" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            BaseConstraint.functional_constraints(
                test_func,
                np.array([0, 1]),
                sample_dataframe,
                X_axis="x_coord",
                Field="field_val",
            )
        assert "Must provide Y_axis as **kwargs" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            BaseConstraint.functional_constraints(
                test_func,
                np.array([0, 1]),
                sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
            )
        assert "Must provide Field as **kwargs" in str(excinfo.value)

    def test_functional_constraints_with_dataframe_z_axis_default(
        self, sample_dataframe
    ):
        """Test functional_constraints with dataframe when Z_axis is not provided."""

        def test_func(x, y, **kwargs):
            return x > y

        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2]), np.array([3, 4]))
            result = BaseConstraint.functional_constraints(
                test_func,
                np.array([0, 1]),
                sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            mock_get_coords.assert_called_once()
            args, kwargs = mock_get_coords.call_args
            np.testing.assert_array_equal(args[0], np.array([0, 1]))
            assert args[1].equals(sample_dataframe)
            assert kwargs["Z_axis"] is None
            assert kwargs["X_axis"] == "x_coord"
            assert kwargs["Y_axis"] == "y_coord"
            assert kwargs["Field"] == "field_val"

            np.testing.assert_array_equal(result, np.array([False, False]))

    def test_get_constraint_indices_with_numpy_array(self, sample_array):
        """Test get_constraint_indices method with numpy array."""

        class MockConstraint(BaseConstraint):
            def constraint_function(self, coords):
                print(f"constraint_function called with coords: {coords}")
                result = coords[0] > coords[1]
                print(f"constraint_function returned: {result}")
                return result

        constraint = MockConstraint(
            data=pd.DataFrame(), X_axis="x", Y_axis="y", Field="f"
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([3, 1, 5]), np.array([2, 4, 3]))
            with patch(
                "pysensors.utils._constraints.BaseConstraint."
                "get_functionalConstraind_sensors_indices"
            ) as mock_get_func_const:
                mock_get_func_const.return_value = (np.array([20]), np.array([0]))
                all_sensors = np.array([10, 20, 30])
                idx_const, rank = constraint.get_constraint_indices(
                    all_sensors, sample_array
                )
                mock_get_coords.assert_called_once_with(all_sensors, sample_array)
                args, _ = mock_get_func_const.call_args
                print(
                    "all_sensors passed to"
                    f"get_functionalConstraind_sensors_indices: {args[0]}"
                )
                print(
                    "g (boolean array) passed to"
                    f"get_functionalConstraind_sensors_indices: {args[1]}"
                )
                np.testing.assert_array_equal(idx_const, np.array([20]))
                np.testing.assert_array_equal(rank, np.array([0]))

    def test_plot_constraint_on_data_with_image(self):
        """Test plot_constraint_on_data method with image plot type."""
        n_features = 256
        n_samples = 10
        sample_data = np.random.rand(n_samples, n_features)

        class MockConstraint(BaseConstraint):
            def draw(self, ax, **kwargs):
                pass

        constraint = MockConstraint(data=sample_data, X_axis="x", Y_axis="y", Field="f")
        with patch.object(constraint, "draw") as mock_draw:
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                constraint.plot_constraint_on_data("image")
                mock_subplots.assert_called_once()
                mock_ax.imshow.assert_called_once()
                args, kwargs = mock_ax.imshow.call_args
                assert args[0].shape == (16, 16)
                assert kwargs["cmap"] == plt.cm.gray
                assert kwargs["interpolation"] == "nearest"
                assert "vmin" in kwargs
                assert "vmax" in kwargs
                mock_draw.assert_called_once_with(
                    mock_ax, alpha=0.3, cmap=plt.cm.coolwarm, s=1, color="red"
                )
        with patch.object(constraint, "draw") as mock_draw:
            existing_fig = MagicMock()
            existing_ax = MagicMock()
            constraint.plot_constraint_on_data(
                "image", plot=(existing_fig, existing_ax)
            )
            assert constraint.fig == existing_fig
            assert constraint.ax == existing_ax
            existing_ax.imshow.assert_called_once()
            mock_draw.assert_called_once_with(
                existing_ax, alpha=0.3, cmap=plt.cm.coolwarm, s=1, color="red"
            )

    def test_get_functionalConstrained_sensors_indices(self):
        """Test get_functionalConstraind_sensors_indices method."""
        senID = np.array([0, 1, 2, 3, 4])
        g = np.array([True, False, True, False, True])
        idx_constrained, rank = BaseConstraint.get_functionalConstraind_sensors_indices(
            senID, g
        )
        np.testing.assert_array_equal(idx_constrained, [1, 3])
        np.testing.assert_array_equal(rank, [0, 1])

    def test_get_constraint_indices(self, sample_dataframe):
        """Test get_constraint_indices method."""

        class MockConstraint(BaseConstraint):
            def constraint_function(self, coords):
                return coords[0] > coords[1]

        constraint = MockConstraint(
            data=sample_dataframe, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2, 3]), np.array([5, 4, 3]))
            all_sensors = np.array([0, 1, 2])
            idx_const, rank = constraint.get_constraint_indices(
                all_sensors, sample_dataframe
            )
            mock_get_coords.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_draw_constraint(self, mock_subplots):
        """Test draw_constraint method."""

        class MockConstraint(BaseConstraint):
            def draw(self, ax, **kwargs):
                pass

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        constraint = MockConstraint(
            data=pd.DataFrame(), X_axis="x", Y_axis="y", Field="f"
        )
        with patch.object(constraint, "draw") as mock_draw:
            constraint.draw_constraint()
            mock_subplots.assert_called_once()
            mock_draw.assert_called_once_with(mock_ax)
            mock_subplots.reset_mock()
            mock_draw.reset_mock()
            existing_plot = (MagicMock(), MagicMock())
            constraint.draw_constraint(plot=existing_plot)
            mock_draw.assert_called_once_with(existing_plot[1])

    @patch("matplotlib.pyplot.subplots")
    def test_plot_constraint_on_data(self, mock_subplots):
        """Test plot_constraint_on_data method."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        class MockConstraint(BaseConstraint):
            def draw(self, ax, **kwargs):
                pass

        df = pd.DataFrame(
            {"x_coord": [1, 2, 3], "y_coord": [3, 2, 1], "field_val": [0.1, 0.2, 0.3]}
        )
        constraint = MockConstraint(
            data=df, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        with patch.object(constraint, "draw") as mock_draw:
            constraint.plot_constraint_on_data("scatter")
            mock_subplots.assert_called_once()
            mock_ax.scatter.assert_called_once()
            mock_draw.assert_called_once()
            mock_subplots.reset_mock()
            mock_ax.scatter.reset_mock()
            mock_draw.reset_mock()
            constraint.plot_constraint_on_data("contour_map")

    @patch("matplotlib.pyplot.subplots")
    def test_plot_grid(self, mock_subplots):
        """Test plot_grid method."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        df = pd.DataFrame(
            {"x_coord": [1, 2, 3], "y_coord": [3, 2, 1], "field_val": [0.1, 0.2, 0.3]}
        )
        constraint = BaseConstraint(
            data=df, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2]), np.array([3, 4]))
            constraint.plot_grid(np.array([0, 1]))
            mock_subplots.assert_called_once()
            mock_ax.scatter.assert_called_once()

    def test_plot_selected_sensors(self, sample_dataframe):
        """Test plot_selected_sensors method."""
        constraint = BaseConstraint(
            data=sample_dataframe, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        constraint.ax = MagicMock()
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.side_effect = [
                (np.array([1, 2]), np.array([3, 4])),
                (np.array([5, 6]), np.array([7, 8])),
            ]
            sensors = np.array([0, 1, 2, 3])
            all_sensors = np.array([2, 3, 0, 1])
            constraint.plot_selected_sensors(sensors, all_sensors)
            constraint.ax.plot.assert_called()

    def test_sensors_dataframe(self, sample_dataframe):
        """Test sensors_dataframe method."""
        constraint = BaseConstraint(
            data=sample_dataframe, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2]), np.array([3, 4]))
            sensors = np.array([0, 1])
            result = constraint.sensors_dataframe(sensors)
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["Sensor ID", "SensorX", "sensorY"]
            assert len(result) == 2

    def test_annotate_sensors(self, sample_dataframe):
        """Test annotate_sensors method."""
        constraint = BaseConstraint(
            data=sample_dataframe, X_axis="x_coord", Y_axis="y_coord", Field="field_val"
        )
        constraint.ax = MagicMock()
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.side_effect = [
                (np.array([1, 2]), np.array([3, 4])),
                (np.array([1]), np.array([3])),
                (np.array([2]), np.array([4])),
            ]
            sensors = np.array([0, 1])
            all_sensors = np.array([1, 0])
            constraint.annotate_sensors(sensors, all_sensors)
            constraint.ax.plot.assert_called()
            constraint.ax.annotate.assert_called()

    def test_plot_selected_sensors_with_numpy_array(self, sample_array):
        """Test plot_selected_sensors method with numpy array data."""
        constraint = BaseConstraint(
            data=sample_array, X_axis="x", Y_axis="y", Field="f"
        )
        constraint.ax = MagicMock()
        sensors = np.array([0, 1, 4, 5, 9])
        all_sensors = np.array([0, 2, 4, 6, 8])
        constraint.plot_selected_sensors(
            sensors=sensors,
            all_sensors=all_sensors,
            color_constrained="blue",
            color_unconstrained="yellow",
        )
        assert constraint.ax.plot.call_count == 2
        first_call_args, first_call_kwargs = constraint.ax.plot.call_args_list[0]
        second_call_args, second_call_kwargs = constraint.ax.plot.call_args_list[1]
        n_features = sample_array.shape[1]
        sqrt_n_features = np.sqrt(n_features)
        expected_constrained = np.array([1, 5, 9])
        expected_unconstrained = np.array([0, 4])
        expected_xconst = np.mod(expected_constrained, sqrt_n_features)
        expected_yconst = np.floor(expected_constrained / sqrt_n_features)
        expected_xunconst = np.mod(expected_unconstrained, sqrt_n_features)
        expected_yunconst = np.floor(expected_unconstrained / sqrt_n_features)
        np.testing.assert_array_equal(first_call_args[0], expected_xconst)
        np.testing.assert_array_equal(first_call_args[1], expected_yconst)
        assert first_call_args[2] == "*"
        assert first_call_kwargs["color"] == "blue"
        np.testing.assert_array_equal(second_call_args[0], expected_xunconst)
        np.testing.assert_array_equal(second_call_args[1], expected_yunconst)
        assert second_call_args[2] == "*"
        assert second_call_kwargs["color"] == "yellow"


class TestCircle:

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init(self, sample_dataframe):
        """Test initialization of Circle class."""
        circle = Circle(
            center_x=0,
            center_y=0,
            radius=2,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        assert circle.center_x == 0
        assert circle.center_y == 0
        assert circle.radius == 2
        assert circle.loc == "in"
        assert circle.data.equals(sample_dataframe)
        assert circle.X_axis == "x_coord"
        assert circle.Y_axis == "y_coord"
        assert circle.Field == "field_val"
        circle_out = Circle(
            center_x=1,
            center_y=2,
            radius=3,
            loc="out",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        assert circle_out.loc == "out"

    def test_draw(self):
        """Test the draw method of the Circle class."""
        mock_ax = MagicMock()
        circle = Circle(
            center_x=1,
            center_y=2,
            radius=3,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        circle.draw(mock_ax)
        mock_ax.add_patch.assert_called_once()
        circle_patch = mock_ax.add_patch.call_args[0][0]
        assert isinstance(circle_patch, patches.Circle)
        assert circle_patch.center == (1, 2)
        assert circle_patch.radius == 3
        assert circle_patch.fill is False
        mock_ax.autoscale_view.assert_called_once()
        mock_ax.reset_mock()
        circle.draw(mock_ax, fill=True, color="blue", lw=3, alpha=0.5)
        mock_ax.add_patch.assert_called_once()
        circle_patch = mock_ax.add_patch.call_args[0][0]
        assert circle_patch.fill is True
        edge_color = circle_patch.get_edgecolor()
        assert edge_color[0] == 0.0
        assert edge_color[1] == 0.0
        assert edge_color[2] > 0.9
        assert circle_patch.get_linewidth() == 3
        assert circle_patch.get_alpha() == 0.5

    def test_constraint_function_inside(self):
        """Test the constraint_function method with 'in' location."""
        circle = Circle(
            center_x=0,
            center_y=0,
            radius=2,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert circle.constraint_function([0, 0]) is False
        assert circle.constraint_function([1, 1]) is False
        assert circle.constraint_function([0, 1.5]) is False

        assert circle.constraint_function([2, 0]) is False
        assert circle.constraint_function([0, 2]) is False

        assert circle.constraint_function([3, 0]) is True
        assert circle.constraint_function([0, 3]) is True
        assert circle.constraint_function([2, 2]) is True

    def test_constraint_function_outside(self):
        """Test the constraint_function method with 'out' location."""
        circle = Circle(
            center_x=0,
            center_y=0,
            radius=2,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert circle.constraint_function([0, 0]) is True
        assert circle.constraint_function([1, 1]) is True
        assert circle.constraint_function([0, 1.5]) is True

        assert circle.constraint_function([2, 0]) is True
        assert circle.constraint_function([0, 2]) is True

        assert circle.constraint_function([3, 0]) is False
        assert circle.constraint_function([0, 3]) is False
        assert circle.constraint_function([2, 2]) is False

    def test_integration_with_base_constraint(self, sample_dataframe):
        """Test that Circle inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(module_path) as mock_get_coords:
            mock_get_coords.return_value = (
                np.array([0, 3]),
                np.array([0, 0]),
            )
            circle = Circle(
                center_x=0,
                center_y=0,
                radius=2,
                loc="in",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_functionalConstraind_sensors_indices"
            ) as mock_get_func:
                mock_get_func.return_value = (
                    [1],
                    [0],
                )
                all_sensors = np.array([0, 1])
                idx_const, rank = circle.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                mock_get_coords.assert_called_once_with(
                    ANY,
                    ANY,
                    X_axis="x_coord",
                    Y_axis="y_coord",
                    Z_axis=None,
                    Field="field_val",
                )
                assert idx_const == [1]
                assert rank == [0]

    def test_with_real_plot(self):
        """Test Circle plot with a real matplotlib figure(visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        df = pd.DataFrame(
            {
                "x_coord": np.random.uniform(-5, 5, 100),
                "y_coord": np.random.uniform(-5, 5, 100),
                "field_val": np.random.random(100),
            }
        )
        circle = Circle(
            center_x=0,
            center_y=0,
            radius=3,
            data=df,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        fig, ax = plt.subplots()
        ax.scatter(df["x_coord"], df["y_coord"], c=df["field_val"], cmap="viridis")
        circle.draw(ax, color="red", lw=2)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Circle Constraint Visualization")
        plt.show()


class TestCylinder:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "z_coord": [10, 20, 30, 40, 50],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init(self, sample_dataframe):
        """Test initialization of Cylinder class."""
        cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=5,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert cylinder.center_x == 0
        assert cylinder.center_y == 0
        assert cylinder.center_z == 0
        assert cylinder.radius == 2
        assert cylinder.height == 5
        assert cylinder.loc == "in"
        assert cylinder.axis == "Z_axis"
        assert cylinder.data.equals(sample_dataframe)
        assert cylinder.X_axis == "x_coord"
        assert cylinder.Y_axis == "y_coord"
        assert cylinder.Field == "field_val"

        cylinder_custom = Cylinder(
            center_x=1,
            center_y=2,
            center_z=3,
            radius=4,
            height=6,
            loc="out",
            axis="X_axis",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )

        assert cylinder_custom.loc == "out"
        assert cylinder_custom.axis == "X_axis"

    def test_draw(self):
        """Test the draw method of the Cylinder class."""
        mock_ax = MagicMock()
        mock_ax.plot_surface = MagicMock()
        cylinder = Cylinder(
            center_x=1,
            center_y=2,
            center_z=3,
            radius=4,
            height=6,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        cylinder.draw(mock_ax)
        assert mock_ax.plot_surface.called
        call_args = mock_ax.plot_surface.call_args
        args, kwargs = call_args
        assert len(args) == 3
        assert all(isinstance(arg, np.ndarray) for arg in args)
        assert "alpha" in kwargs
        assert "color" in kwargs
        assert kwargs["color"] == "red"
        assert mock_ax.autoscale_view.called
        mock_ax.reset_mock()
        cylinder.draw(mock_ax, alpha=0.5, color="blue")
        assert mock_ax.plot_surface.called
        _, kwargs = mock_ax.plot_surface.call_args
        assert kwargs["color"] == "blue"
        assert "alpha" in kwargs

    def test_axis_parameter(self, sample_dataframe):
        """Test that cylinder can be correctly oriented along different axes."""
        z_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert z_cylinder.axis == "Z_axis"
        x_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            axis="X_axis",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert x_cylinder.axis == "X_axis"
        y_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            axis="Y_axis",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert y_cylinder.axis == "Y_axis"

    def test_in_out_parameter(self, sample_dataframe):
        """Test that cylinder respects the 'in'/'out' parameter."""
        in_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert in_cylinder.loc == "in"

        out_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            loc="out",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        assert out_cylinder.loc == "out"

    def test_constraint_function_general(self):
        """Test basic functionality of the constraint_function method."""
        z_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )

        x_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            axis="X_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )

        y_cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=6,
            axis="Y_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )

        assert z_cylinder.axis == "Z_axis"
        assert x_cylinder.axis == "X_axis"
        assert y_cylinder.axis == "Y_axis"

    def test_integration_with_base_constraint(self, sample_dataframe):
        """Test that Cylinder inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(
            module_path,
            return_value=(
                np.array([0, 3, 0]),
                np.array([0, 0, 0]),
                np.array([0, 0, 4]),
            ),
        ):
            cylinder = Cylinder(
                center_x=0,
                center_y=0,
                center_z=0,
                radius=2,
                height=6,
                loc="in",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Z_axis="z_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_constraint_indices", return_value=([1, 2], [0, 1])
            ):
                all_sensors = np.array([0, 1, 2])
                idx_const, rank = cylinder.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                assert idx_const == [1, 2]
                assert rank == [0, 1]

    def test_with_real_plot(self):
        """Test Cylinder plot with a real matplotlib figure (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        df = pd.DataFrame(
            {
                "x_coord": np.random.uniform(-5, 5, 100),
                "y_coord": np.random.uniform(-5, 5, 100),
                "z_coord": np.random.uniform(-5, 5, 100),
                "field_val": np.random.random(100),
            }
        )
        cylinder = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=4,
            data=df,
            X_axis="x_coord",
            Y_axis="y_coord",
            Z_axis="z_coord",
            Field="field_val",
        )
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df["x_coord"],
            df["y_coord"],
            df["z_coord"],
            c=df["field_val"],
            cmap="viridis",
            alpha=0.7,
        )
        cylinder.draw(ax, color="red", alpha=0.3)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")
        ax.set_title("Cylinder Constraint Visualization")
        ax.view_init(elev=30, azim=45)
        plt.show()

    def test_draw_axis_options(self):
        """Test the draw method of Cylinder with different axis orientations."""
        mock_ax = MagicMock()
        z_cylinder = Cylinder(
            center_x=1,
            center_y=2,
            center_z=3,
            radius=2,
            height=4,
            axis="Z_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        z_cylinder.draw(mock_ax)
        assert mock_ax.plot_surface.called
        args, _ = mock_ax.plot_surface.call_args
        x, y, z = args
        assert np.min(z) < z_cylinder.center_z - z_cylinder.height / 3
        assert np.max(z) > z_cylinder.center_z + z_cylinder.height / 3
        mock_ax.reset_mock()
        x_cylinder = Cylinder(
            center_x=1,
            center_y=2,
            center_z=3,
            radius=2,
            height=4,
            axis="X_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        x_cylinder.draw(mock_ax)
        assert mock_ax.plot_surface.called
        args, _ = mock_ax.plot_surface.call_args
        x, y, z = args
        assert np.min(x) < x_cylinder.center_x - x_cylinder.height / 3
        assert np.max(x) > x_cylinder.center_x + x_cylinder.height / 3
        mock_ax.reset_mock()
        y_cylinder = Cylinder(
            center_x=1,
            center_y=2,
            center_z=3,
            radius=2,
            height=4,
            axis="Y_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        y_cylinder.draw(mock_ax)
        assert mock_ax.plot_surface.called
        args, _ = mock_ax.plot_surface.call_args
        x, y, z = args
        assert np.min(y) < y_cylinder.center_y - y_cylinder.height / 3
        assert np.max(y) > y_cylinder.center_y + y_cylinder.height / 3
        x_min, x_max = np.min(x), np.max(x)
        z_min, z_max = np.min(z), np.max(z)
        assert x_min < y_cylinder.center_x - y_cylinder.radius / 2
        assert x_max > y_cylinder.center_x + y_cylinder.radius / 2
        assert z_min < y_cylinder.center_z - y_cylinder.radius / 2
        assert z_max > y_cylinder.center_z + y_cylinder.radius / 2

    def test_constraint_function(self):
        """Test constraint_function method of Cylinder class for all configurations."""

        z_cylinder_in = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=4,
            loc="in",
            axis="Z_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        y_cylinder_in = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=4,
            loc="in",
            axis="Y_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        x_cylinder_in = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=4,
            loc="in",
            axis="X_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        z_cylinder_out = Cylinder(
            center_x=0,
            center_y=0,
            center_z=0,
            radius=2,
            height=4,
            loc="out",
            axis="Z_axis",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Z_axis="z",
            Field="f",
        )
        coords_inside_z = np.array([0.0, 0.0, 0.0])
        result = z_cylinder_in.constraint_function(coords_inside_z)
        assert list(result) == [False]
        coords_outside_z_radius = np.array([3.0, 0.0, 0.0])
        result = z_cylinder_in.constraint_function(coords_outside_z_radius)
        assert list(result) == [True]
        coords_outside_z_height = np.array([0.0, 0.0, 3.0])
        result = z_cylinder_in.constraint_function(coords_outside_z_height)
        assert list(result) == [True]
        coords_multiple_z = np.array(
            [[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
        ).T
        result = z_cylinder_in.constraint_function(coords_multiple_z)
        assert list(result) == [False, True, True, True]
        coords_inside_y = np.array([0.0, 0.0, 0.0])
        result = y_cylinder_in.constraint_function(coords_inside_y)
        assert list(result) == [False]
        coords_outside_y_radius = np.array([3.0, 0.0, 0.0])
        result = y_cylinder_in.constraint_function(coords_outside_y_radius)
        assert list(result) == [True]
        coords_outside_y_height = np.array([0.0, 3.0, 0.0])
        result = y_cylinder_in.constraint_function(coords_outside_y_height)
        assert list(result) == [True]
        coords_inside_x = np.array([0.0, 0.0, 0.0])
        result = x_cylinder_in.constraint_function(coords_inside_x)
        assert list(result) == [False]
        coords_outside_x_radius = np.array([0.0, 3.0, 0.0])
        result = x_cylinder_in.constraint_function(coords_outside_x_radius)
        assert list(result) == [True]
        coords_outside_x_height = np.array([3.0, 0.0, 0.0])
        result = x_cylinder_in.constraint_function(coords_outside_x_height)
        assert list(result) == [True]
        result_inside = z_cylinder_out.constraint_function(coords_inside_z)
        result_outside_radius = z_cylinder_out.constraint_function(
            coords_outside_z_radius
        )
        result_outside_height = z_cylinder_out.constraint_function(
            coords_outside_z_height
        )
        result_multiple = z_cylinder_out.constraint_function(coords_multiple_z)
        assert list(result_inside) == [True]
        assert list(result_outside_radius) == [False]
        assert list(result_outside_height) == [False]
        assert list(result_multiple) == [True, False, False, False]


class TestLine:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init(self, sample_dataframe):
        """Test initialization of Line class."""
        line = Line(
            x1=0,
            y1=0,
            x2=5,
            y2=5,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        assert line.x1 == 0
        assert line.y1 == 0
        assert line.x2 == 5
        assert line.y2 == 5
        assert line.data.equals(sample_dataframe)
        assert line.X_axis == "x_coord"
        assert line.Y_axis == "y_coord"
        assert line.Field == "field_val"
        line_horizontal = Line(
            x1=0,
            y1=3,
            x2=5,
            y2=3,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )

        assert line_horizontal.x1 == 0
        assert line_horizontal.y1 == 3
        assert line_horizontal.x2 == 5
        assert line_horizontal.y2 == 3
        line_vertical = Line(
            x1=3,
            y1=0,
            x2=3,
            y2=5,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )

        assert line_vertical.x1 == 3
        assert line_vertical.y1 == 0
        assert line_vertical.x2 == 3
        assert line_vertical.y2 == 5

    def test_draw(self):
        """Test the draw method of the Line class."""
        mock_ax = MagicMock()
        mock_ax.plot = MagicMock()
        line = Line(
            x1=0,
            y1=0,
            x2=5,
            y2=5,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        line.draw(mock_ax)
        assert mock_ax.plot.called
        call_args = mock_ax.plot.call_args
        args, kwargs = call_args
        assert len(args) == 2
        assert args[0] == [0, 5]
        assert args[1] == [0, 5]
        assert kwargs["color"] == "r"
        assert kwargs["alpha"] == 1.0
        assert kwargs["marker"] is None
        assert kwargs["linestyle"] == "-"
        mock_ax.reset_mock()
        line.draw(mock_ax, color="blue", lw=3, alpha=0.5, marker="o", linestyle="--")
        assert mock_ax.plot.called
        _, kwargs = mock_ax.plot.call_args
        assert kwargs["color"] == "blue"
        assert kwargs["alpha"] == 0.5
        assert kwargs["marker"] == "o"
        assert kwargs["linestyle"] == "--"

    def test_constraint_function(self):
        """Test the constraint_function method with various points."""
        line = Line(
            x1=0,
            y1=0,
            x2=5,
            y2=5,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        point_on_line = [2, 2]
        result = line.constraint_function(point_on_line)
        assert result is True
        point_above = [2, 3]
        result = line.constraint_function(point_above)
        assert result is True
        point_below = [3, 2]
        result = line.constraint_function(point_below)
        assert result is False
        line_horizontal = Line(
            x1=0,
            y1=3,
            x2=5,
            y2=3,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        result = line_horizontal.constraint_function([2, 4])
        assert result is True
        result = line_horizontal.constraint_function([2, 2])
        assert result is False

    def test_multiple_points_constraint(self):
        """Test constraint function with multiple points at once."""
        line = Line(
            x1=0,
            y1=0,
            x2=5,
            y2=5,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )

        with patch.object(line, "constraint_function") as mock_func:
            mock_func.side_effect = [True, False, True]
            point1 = [1, 2]
            point2 = [2, 1]
            point3 = [3, 4]

            result1 = line.constraint_function(point1)
            result2 = line.constraint_function(point2)
            result3 = line.constraint_function(point3)

            assert result1 is True
            assert result2 is False
            assert result3 is True

    def test_integration_with_base_constraint(self, sample_dataframe):
        """Test that Line inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(
            module_path,
            return_value=(
                np.array([1, 3, 2]),
                np.array([1, 4, 1]),
            ),
        ):
            line = Line(
                x1=0,
                y1=0,
                x2=5,
                y2=5,
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_constraint_indices", return_value=([1], [0])
            ):
                all_sensors = np.array([0, 1, 2])
                idx_const, rank = line.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                assert idx_const == [1]
                assert rank == [0]

    def test_with_real_plot(self):
        """Test Line plotting with a real matplotlib figure (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        df = pd.DataFrame(
            {
                "x_coord": np.random.uniform(-5, 5, 100),
                "y_coord": np.random.uniform(-5, 5, 100),
                "field_val": np.random.random(100),
            }
        )
        line = Line(
            x1=-4,
            y1=-3,
            x2=4,
            y2=3,
            data=df,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            df["x_coord"], df["y_coord"], c=df["field_val"], cmap="viridis", alpha=0.7
        )
        line.draw(ax, color="red", lw=2)
        plt.colorbar(scatter, label="Field Value")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Line Constraint Visualization")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.show()

    def test_special_cases(self):
        """Test Line with special cases like horizontal and vertical lines."""
        horizontal_line = Line(
            x1=-5,
            y1=0,
            x2=5,
            y2=0,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )

        vertical_line = Line(
            x1=0,
            y1=-5,
            x2=0,
            y2=5,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert horizontal_line.constraint_function([0, 1]) is True
        assert horizontal_line.constraint_function([0, -1]) is False
        assert vertical_line.constraint_function([-1, 0]) is True
        assert vertical_line.constraint_function([1, 0]) is False


class TestParabola:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init(self, sample_dataframe):
        """Test initialization of Parabola class."""
        parabola_in = Parabola(
            h=0,
            k=0,
            a=1,
            loc="in",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        assert parabola_in.h == 0
        assert parabola_in.k == 0
        assert parabola_in.a == 1
        assert parabola_in.loc == "in"
        assert parabola_in.data.equals(sample_dataframe)
        assert parabola_in.X_axis == "x_coord"
        assert parabola_in.Y_axis == "y_coord"
        assert parabola_in.Field == "field_val"
        parabola_out = Parabola(
            h=2,
            k=3,
            a=0.5,
            loc="out",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )

        assert parabola_out.h == 2
        assert parabola_out.k == 3
        assert parabola_out.a == 0.5
        assert parabola_out.loc == "out"

    def test_draw(self):
        """Test the draw method of the Parabola class."""
        mock_ax = MagicMock()
        mock_ax.scatter = MagicMock()
        parabola_np = Parabola(
            h=0,
            k=0,
            a=1,
            loc="in",
            data=np.zeros((5, 25)),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            x_coords = np.array([0, 1, 2, 3, 4])
            y_coords = np.array([0, 1, 2, 3, 4])
            mock_get_coords.return_value = (x_coords, y_coords)
            parabola_np.draw(mock_ax)
            mock_get_coords.assert_called_once()
            assert mock_ax.scatter.called
            call_args = mock_ax.scatter.call_args
            args, kwargs = call_args
            assert len(args) == 2
            assert np.array_equal(args[0], x_coords)
            expected_y = (
                parabola_np.a * ((x_coords - parabola_np.h) ** 2) - parabola_np.k
            )
            assert np.array_equal(args[1], expected_y)

    def test_draw_with_dataframe(self, sample_dataframe):
        """Test the draw method with a pandas DataFrame."""
        mock_ax = MagicMock()
        mock_ax.scatter = MagicMock()
        parabola_df = Parabola(
            h=1,
            k=2,
            a=0.5,
            loc="in",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            x_coords = np.array([1, 2, 3, 4, 5])
            y_coords = np.array([5, 4, 3, 2, 1])
            mock_get_coords.return_value = (x_coords, y_coords)
            parabola_df.draw(mock_ax)
            mock_get_coords.assert_called_once()
            call_args = mock_get_coords.call_args
            args, kwargs = call_args
            assert np.array_equal(args[0], np.arange(len(sample_dataframe)))
            assert args[1] is sample_dataframe
            assert kwargs["X_axis"] == "x_coord"
            assert kwargs["Y_axis"] == "y_coord"
            assert kwargs["Field"] == "field_val"
            assert mock_ax.scatter.called

    def test_constraint_function_opening_up(self):
        """Test the constraint_function method with a parabola opening upward."""
        parabola = Parabola(
            h=0,
            k=0,
            a=1,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        point_inside = [0, 1]
        result = parabola.constraint_function(point_inside)
        assert result is False
        point_outside = [2, 1]
        result = parabola.constraint_function(point_outside)
        assert result is True
        point_on = [1, 1]
        result = parabola.constraint_function(point_on)
        assert result is False
        parabola_out = Parabola(
            h=0,
            k=0,
            a=1,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert parabola_out.constraint_function(point_inside) is True
        assert parabola_out.constraint_function(point_outside) is False
        assert parabola_out.constraint_function(point_on) is True

    def test_constraint_function_opening_down(self):
        """Test the constraint_function method with a parabola opening downward."""
        parabola = Parabola(
            h=0,
            k=0,
            a=-1,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        point_inside = [0, -1]
        result = parabola.constraint_function(point_inside)
        assert result is True
        point_outside = [2, 0]
        result = parabola.constraint_function(point_outside)
        assert result is False

    def test_constraint_function_shifted_parabola(self):
        """Test the constraint_function with a shifted parabola."""
        parabola = Parabola(
            h=2,
            k=3,
            a=0.5,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        point_inside = [2, 4]
        result = parabola.constraint_function(point_inside)
        assert result is False
        point_outside = [4, 3]
        result = parabola.constraint_function(point_outside)
        assert result is True
        point_on = [4, 5]
        result = parabola.constraint_function(point_on)
        assert result is False

    def test_integration_with_base_constraint(self, sample_dataframe):
        """Test that Parabola inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(
            module_path,
            return_value=(
                np.array([0, 2, 4]),
                np.array([2, 0, 6]),
            ),
        ):
            parabola = Parabola(
                h=0,
                k=0,
                a=1,
                loc="in",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_constraint_indices", return_value=([1, 2], [0, 1])
            ):
                all_sensors = np.array([0, 1, 2])
                idx_const, rank = parabola.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                assert idx_const == [1, 2]
                assert rank == [0, 1]

    def test_with_real_plot(self):
        """Test Parabola plot with a real matplotlib figure (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")

        n_points = 100
        df = pd.DataFrame(
            {
                "x_coord": np.linspace(-5, 5, n_points),
                "y_coord": np.random.uniform(-2, 8, n_points),
                "field_val": np.random.random(n_points),
            }
        )
        parabola = Parabola(
            h=0,
            k=0,
            a=0.5,
            loc="in",
            data=df,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df["x_coord"], df["y_coord"], c=df["field_val"], cmap="viridis", alpha=0.7
        )
        parabola.draw(ax)
        plt.colorbar(scatter, label="Field Value")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Parabola Constraint Visualization")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 8)
        plt.show()


class TestEllipse:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def sample_array(self):
        """Create a sample numpy array for testing."""
        return np.random.rand(5, 16)

    def test_init(self, sample_dataframe):
        """Test initialization of Ellipse class."""
        ellipse = Ellipse(
            center_x=0,
            center_y=0,
            width=4,
            height=2,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        assert ellipse.center_x == 0
        assert ellipse.center_y == 0
        assert ellipse.width == 4
        assert ellipse.height == 2
        assert ellipse.angle == 0.0
        assert ellipse.loc == "in"
        assert ellipse.half_horizontal_axis == 2
        assert ellipse.half_vertical_axis == 1
        assert ellipse.data.equals(sample_dataframe)
        assert ellipse.X_axis == "x_coord"
        assert ellipse.Y_axis == "y_coord"
        assert ellipse.Field == "field_val"
        ellipse_custom = Ellipse(
            center_x=1,
            center_y=2,
            width=6,
            height=4,
            angle=45.0,
            loc="out",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )

        assert ellipse_custom.center_x == 1
        assert ellipse_custom.center_y == 2
        assert ellipse_custom.width == 6
        assert ellipse_custom.height == 4
        assert ellipse_custom.angle == 45.0
        assert ellipse_custom.loc == "out"
        assert ellipse_custom.half_horizontal_axis == 3
        assert ellipse_custom.half_vertical_axis == 2

    def test_draw(self):
        """Test the draw method of the Ellipse class."""
        mock_ax = MagicMock()
        mock_ax.add_patch = MagicMock()
        ellipse = Ellipse(
            center_x=0,
            center_y=0,
            width=4,
            height=2,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        ellipse.draw(mock_ax)
        assert mock_ax.add_patch.called
        call_args = mock_ax.add_patch.call_args
        args, _ = call_args
        assert isinstance(args[0], patches.Ellipse)
        ellipse_patch = args[0]
        assert ellipse_patch.center == (0, 0)
        assert ellipse_patch.width == 4
        assert ellipse_patch.height == 2
        assert ellipse_patch.angle == 0.0
        assert ellipse_patch.fill is False
        assert ellipse_patch.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)
        assert mock_ax.autoscale_view.called
        mock_ax.reset_mock()
        ellipse.draw(mock_ax, fill=True, color="blue", lw=3, alpha=0.5)
        assert mock_ax.add_patch.called
        ellipse_patch = mock_ax.add_patch.call_args[0][0]
        assert ellipse_patch.fill is True
        assert ellipse_patch.get_edgecolor()[2] > 0.5
        assert ellipse_patch.get_edgecolor()[0] < 0.5
        assert ellipse_patch.get_edgecolor()[1] < 0.5
        assert ellipse_patch.get_alpha() == 0.5

    def test_constraint_function_circle(self):
        """Test the constraint_function with a circle (special case of ellipse)."""
        circle = Ellipse(
            center_x=0,
            center_y=0,
            width=4,
            height=4,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert circle.constraint_function([0, 0]) is False
        assert circle.constraint_function([1, 1]) is False
        assert circle.constraint_function([2, 0]) is False
        assert circle.constraint_function([3, 0]) is True
        circle_out = Ellipse(
            center_x=0,
            center_y=0,
            width=4,
            height=4,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert circle_out.constraint_function([0, 0]) == True  # noqa:E712
        assert circle_out.constraint_function([1, 1]) == True  # noqa:E712
        assert circle_out.constraint_function([2, 0]) == True  # noqa:E712
        assert circle_out.constraint_function([3, 0]) == False  # noqa:E712

    def test_constraint_function_ellipse(self):
        """Test the constraint_function with a proper ellipse."""
        ellipse = Ellipse(
            center_x=0,
            center_y=0,
            width=6,
            height=4,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert ellipse.constraint_function([0, 0]) is False
        assert ellipse.constraint_function([2, 0]) is False
        assert ellipse.constraint_function([0, 1]) is False
        assert ellipse.constraint_function([3, 0]) is False
        assert ellipse.constraint_function([0, 2]) is False
        assert ellipse.constraint_function([4, 0]) is True
        assert ellipse.constraint_function([0, 3]) is True
        assert ellipse.constraint_function([2, 1.5]) is True

    def test_constraint_function_rotated_ellipse(self):
        """Test the constraint_function with a rotated ellipse."""
        ellipse_rotated = Ellipse(
            center_x=0,
            center_y=0,
            width=6,
            height=4,
            angle=45.0,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert ellipse_rotated.constraint_function([0, 0]) is False
        assert ellipse_rotated.constraint_function([1.5, 1.5]) is False
        assert ellipse_rotated.constraint_function([4, 4]) is True

    def test_constraint_function_shifted_ellipse(self):
        """Test the constraint_function with a shifted ellipse."""
        ellipse_shifted = Ellipse(
            center_x=3,
            center_y=2,
            width=6,
            height=4,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert ellipse_shifted.constraint_function([3, 2]) is False
        assert ellipse_shifted.constraint_function([5, 2]) is False
        assert ellipse_shifted.constraint_function([3, 3]) is False
        assert ellipse_shifted.constraint_function([6, 2]) is False
        assert ellipse_shifted.constraint_function([3, 4]) is False
        assert ellipse_shifted.constraint_function([7, 2]) is True
        assert ellipse_shifted.constraint_function([3, 5]) is True

    def test_integration_with_base_constraint(self, sample_dataframe):
        """Test that Ellipse inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(
            module_path,
            return_value=(
                np.array([0, 3, 2]),
                np.array([0, 0, 3]),
            ),
        ):
            ellipse = Ellipse(
                center_x=0,
                center_y=0,
                width=4,
                height=2,
                loc="in",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_constraint_indices", return_value=([1, 2], [0, 1])
            ):
                all_sensors = np.array([0, 1, 2])
                idx_const, rank = ellipse.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                assert idx_const == [1, 2]
                assert rank == [0, 1]

    def test_with_real_plot(self):
        """Test Ellipse plot with a real matplotlib figure (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        n_points = 200
        df = pd.DataFrame(
            {
                "x_coord": np.random.uniform(-5, 5, n_points),
                "y_coord": np.random.uniform(-5, 5, n_points),
                "field_val": np.random.random(n_points),
            }
        )
        ellipses = [
            Ellipse(
                center_x=0,
                center_y=0,
                width=6,
                height=4,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            Ellipse(
                center_x=0,
                center_y=0,
                width=6,
                height=4,
                angle=45.0,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            Ellipse(
                center_x=2,
                center_y=-2,
                width=4,
                height=4,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (ax, ellipse) in enumerate(zip(axes, ellipses)):
            scatter = ax.scatter(
                df["x_coord"],
                df["y_coord"],
                c=df["field_val"],
                cmap="viridis",
                alpha=0.7,
                s=10,
            )
            ellipse.draw(ax, color=["r", "g", "b"][i], lw=2)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_title(
                ["Standard Ellipse", "Rotated Ellipse (45°)", "Shifted Ellipse"][i]
            )
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label="Field Value")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()


class TestPolygon:

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def triangle_coords(self):
        """Create coordinates for a triangle."""
        return np.array([[0, 0], [4, 0], [2, 4]])

    @pytest.fixture
    def square_coords(self):
        """Create coordinates for a square."""
        return np.array([[0, 0], [4, 0], [4, 4], [0, 4]])

    @pytest.fixture
    def pentagon_coords(self):
        """Create coordinates for a pentagon."""
        return np.array([[0, 0], [4, 0], [5, 3], [2, 5], [-1, 3]])

    def test_init(self, sample_dataframe, triangle_coords):
        """Test initialization of Polygon class."""
        polygon = Polygon(
            xy_coords=triangle_coords,
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        np.testing.assert_array_equal(polygon.xy_coords, triangle_coords)
        assert polygon.loc == "in"
        assert polygon.data.equals(sample_dataframe)
        assert polygon.X_axis == "x_coord"
        assert polygon.Y_axis == "y_coord"
        assert polygon.Field == "field_val"
        polygon_out = Polygon(
            xy_coords=triangle_coords,
            loc="out",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )

        np.testing.assert_array_equal(polygon_out.xy_coords, triangle_coords)
        assert polygon_out.loc == "out"

    def test_draw(self, triangle_coords):
        """Test the draw method of the Polygon class."""
        mock_ax = MagicMock()
        mock_ax.add_patch = MagicMock()
        polygon = Polygon(
            xy_coords=triangle_coords,
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        polygon.draw(mock_ax)
        assert mock_ax.add_patch.called
        call_args = mock_ax.add_patch.call_args
        args, _ = call_args
        assert isinstance(args[0], patches.Polygon)
        polygon_patch = args[0]
        xy = polygon_patch.get_xy()
        for i, coord in enumerate(triangle_coords):
            assert np.array_equal(xy[i], coord)
        assert polygon_patch.fill is False
        assert polygon_patch.get_edgecolor()[0] == 1.0
        assert mock_ax.autoscale_view.called
        mock_ax.reset_mock()
        polygon.draw(mock_ax, fill=True, color="blue", lw=3, alpha=0.5)
        assert mock_ax.add_patch.called
        polygon_patch = mock_ax.add_patch.call_args[0][0]
        assert polygon_patch.fill is True
        assert polygon_patch.get_edgecolor()[0] < 0.5
        assert polygon_patch.get_alpha() == 0.5

    def test_constraint_function_triangle(self, triangle_coords):
        """Test the constraint_function with a triangle."""
        triangle_in = Polygon(
            xy_coords=triangle_coords,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert triangle_in.constraint_function([2, 1]) is False
        assert triangle_in.constraint_function([1, 1]) is False
        assert triangle_in.constraint_function([3, 1]) is False
        assert triangle_in.constraint_function([2, 2]) is False
        assert triangle_in.constraint_function([-1, 0]) is True
        assert triangle_in.constraint_function([5, 0]) is True
        assert triangle_in.constraint_function([2, 5]) is True
        assert triangle_in.constraint_function([0, -1]) is True
        triangle_out = Polygon(
            xy_coords=triangle_coords,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert triangle_out.constraint_function([2, 1]) is True
        assert triangle_out.constraint_function([1, 1]) is True
        assert triangle_out.constraint_function([3, 1]) is True
        assert triangle_out.constraint_function([-1, 0]) is False
        assert triangle_out.constraint_function([5, 0]) is False
        assert triangle_out.constraint_function([2, 5]) is False

    def test_constraint_function_square(self, square_coords):
        """Test the constraint_function with a square."""
        square = Polygon(
            xy_coords=square_coords,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert square.constraint_function([2, 2]) is False
        assert square.constraint_function([1, 1]) is False
        assert square.constraint_function([3, 3]) is False
        assert square.constraint_function([-1, 2]) is True
        assert square.constraint_function([5, 2]) is True
        assert square.constraint_function([2, -1]) is True
        assert square.constraint_function([2, 5]) is True
        assert square.constraint_function([0, 0]) is False
        square_out = Polygon(
            xy_coords=square_coords,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert square_out.constraint_function([2, 2]) is True
        assert square_out.constraint_function([1, 1]) is True
        assert square_out.constraint_function([3, 3]) is True
        assert square_out.constraint_function([-1, 2]) is False
        assert square_out.constraint_function([5, 2]) is False
        assert square_out.constraint_function([2, -1]) is False
        assert square_out.constraint_function([2, 5]) is False
        assert square_out.constraint_function([0, 0]) is True

    def test_constraint_function_concave_polygon(self, pentagon_coords):
        """Test the constraint_function with a concave polygon."""
        pentagon = Polygon(
            xy_coords=pentagon_coords,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert pentagon.constraint_function([2, 2]) is False
        assert pentagon.constraint_function([1, 1]) is False
        assert pentagon.constraint_function([3, 3]) is False

        assert pentagon.constraint_function([-2, 2]) is True
        assert pentagon.constraint_function([6, 2]) is True
        assert pentagon.constraint_function([2, -1]) is True
        assert pentagon.constraint_function([2, 6]) is True
        result_00 = pentagon.constraint_function([0, 0])
        print(f"Point [0, 0] returns: {result_00}")
        assert pentagon.constraint_function([0, 0]) is False
        pentagon_out = Polygon(
            xy_coords=pentagon_coords,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert pentagon_out.constraint_function([2, 2]) is True
        assert pentagon_out.constraint_function([1, 1]) is True
        assert pentagon_out.constraint_function([3, 3]) is True
        assert pentagon_out.constraint_function([-2, 2]) is False
        assert pentagon_out.constraint_function([6, 2]) is False
        assert pentagon_out.constraint_function([2, -1]) is False
        assert pentagon_out.constraint_function([2, 6]) is False
        assert pentagon_out.constraint_function([0, 0]) is True

    def test_complex_polygons(self):
        """Test the constraint function with more complex polygons."""
        star_coords = np.array(
            [
                [3, 0],
                [4, 2],
                [6, 2],
                [5, 3],
                [6, 5],
                [3, 4],
                [0, 5],
                [1, 3],
                [0, 2],
                [2, 2],
            ]
        )
        star = Polygon(
            xy_coords=star_coords,
            loc="in",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert star.constraint_function([3, 3]) is False
        assert star.constraint_function([3, 1]) is False
        assert star.constraint_function([5, 3]) is True
        assert star.constraint_function([-2, 3]) is True
        assert star.constraint_function([8, 3]) is True
        assert star.constraint_function([3, -2]) is True
        assert star.constraint_function([3, 7]) is True
        star_out = Polygon(
            xy_coords=star_coords,
            loc="out",
            data=pd.DataFrame(),
            X_axis="x",
            Y_axis="y",
            Field="f",
        )
        assert star_out.constraint_function([3, 3]) is True
        assert star_out.constraint_function([3, 1]) is True
        assert star_out.constraint_function([5, 3]) is False
        assert star_out.constraint_function([-2, 3]) is False
        assert star_out.constraint_function([8, 3]) is False
        assert star_out.constraint_function([3, -2]) is False
        assert star_out.constraint_function([3, 7]) is False

    def test_integration_with_base_constraint(self, sample_dataframe, triangle_coords):
        """Test that Polygon inherits and works with BaseConstraint methods."""
        module_path = BaseConstraint.__module__ + ".get_coordinates_from_indices"

        with patch(
            module_path,
            return_value=(
                np.array([1, 5, 2]),
                np.array([1, 1, 3]),
            ),
        ):
            polygon = Polygon(
                xy_coords=triangle_coords,
                loc="in",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "get_constraint_indices", return_value=([1, 2], [0, 1])
            ):
                all_sensors = np.array([0, 1, 2])
                idx_const, rank = polygon.get_constraint_indices(
                    all_sensors, sample_dataframe
                )
                assert idx_const == [1, 2]
                assert rank == [0, 1]

    def test_with_real_plot(self, triangle_coords, square_coords, pentagon_coords):
        """Test Polygon plot with a real matplotlib figure (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        n_points = 200
        df = pd.DataFrame(
            {
                "x_coord": np.random.uniform(-2, 6, n_points),
                "y_coord": np.random.uniform(-2, 6, n_points),
                "field_val": np.random.random(n_points),
            }
        )
        polygons = [
            Polygon(
                xy_coords=triangle_coords,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            Polygon(
                xy_coords=square_coords,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            Polygon(
                xy_coords=pentagon_coords,
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (ax, polygon) in enumerate(zip(axes, polygons)):
            scatter = ax.scatter(
                df["x_coord"],
                df["y_coord"],
                c=df["field_val"],
                cmap="viridis",
                alpha=0.7,
                s=10,
            )
            polygon.draw(ax, color=["r", "g", "b"][i], lw=2)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_title(["Triangle", "Square", "Pentagon"][i])
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlim(-2, 6)
            ax.set_ylim(-2, 6)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label="Field Value")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()


class TestUserDefinedConstraints:

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "x_coord": [1, 2, 3, 4, 5],
                "y_coord": [5, 4, 3, 2, 1],
                "field_val": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    @pytest.fixture
    def all_sensors(self):
        """Create a sample list of sensors."""
        return np.array([0, 1, 2, 3, 4])

    def test_init_with_equation(self, sample_dataframe, all_sensors):
        """Test initialization with equation parameter."""
        constraint = UserDefinedConstraints(
            all_sensors=all_sensors,
            equation="x**2 + y**2 <= 4",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        np.testing.assert_array_equal(constraint.all_sensors, all_sensors)
        assert constraint.equations == ["x**2 + y**2 <= 4"]
        assert constraint.file is None
        assert constraint.data.equals(sample_dataframe)
        assert constraint.X_axis == "x_coord"
        assert constraint.Y_axis == "y_coord"
        assert constraint.Field == "field_val"

    def test_init_with_file(self, sample_dataframe, all_sensors):
        """Test initialization with file parameter."""
        with patch(
            "pysensors.utils._constraints.load_functional_constraints"
        ) as mock_load:
            mock_load.return_value = lambda x, y: x**2 + y**2 - 4
            constraint = UserDefinedConstraints(
                all_sensors=all_sensors,
                file="test_constraint.py",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            np.testing.assert_array_equal(constraint.all_sensors, all_sensors)
            assert constraint.file == "test_constraint.py"
            assert constraint.equations is None
            assert callable(constraint.functions)
            assert constraint.data.equals(sample_dataframe)
            assert constraint.X_axis == "x_coord"
            assert constraint.Y_axis == "y_coord"
            assert constraint.Field == "field_val"
            mock_load.assert_called_once_with("test_constraint.py")

    def test_init_exceptions(self, all_sensors, sample_dataframe):
        """Test initialization exceptions."""
        with pytest.raises(Exception) as excinfo:
            UserDefinedConstraints(
                all_sensors=all_sensors,
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
        assert "either file or equation should be provided" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            UserDefinedConstraints(
                all_sensors=all_sensors,
                equation="x**2 + y**2 <= 4",
                data=sample_dataframe,
                Y_axis="y_coord",
                Field="field_val",
            )
        assert "Must provide X_axis" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            UserDefinedConstraints(
                all_sensors=all_sensors,
                equation="x**2 + y**2 <= 4",
                data=sample_dataframe,
                X_axis="x_coord",
                Field="field_val",
            )
        assert "Must provide Y_axis" in str(excinfo.value)
        with pytest.raises(Exception) as excinfo:
            UserDefinedConstraints(
                all_sensors=all_sensors,
                equation="x**2 + y**2 <= 4",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
            )
        assert "Must provide" in str(excinfo.value)

    def test_draw_with_equation(self, all_sensors, sample_dataframe):
        """Test the draw method with equation-based constraint."""
        constraint = UserDefinedConstraints(
            all_sensors=all_sensors,
            equation="x**2 + y**2 <= 4",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        mock_ax = MagicMock()
        mock_ax.scatter = MagicMock()
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.side_effect = [
                (
                    np.array([1, 2, 3]),
                    np.array([5, 4, 3]),
                ),
                (
                    np.array([1, 2]),
                    np.array([5, 4]),
                ),
            ]
            with patch.object(
                BaseConstraint, "get_functionalConstraind_sensors_indices"
            ) as mock_get_func:
                mock_get_func.return_value = (
                    [0, 1],
                    [0, 1],
                )
                constraint.draw(mock_ax)
                assert mock_get_coords.call_count == 2
                assert mock_ax.scatter.called
                call_args = mock_ax.scatter.call_args
                args, kwargs = call_args
                assert len(args) == 2
                assert args[0].shape == (2,)
                assert args[1].shape == (2,)
                assert kwargs["s"] == 1

    def test_draw_with_file(self, all_sensors, sample_dataframe):
        """Test the draw method with file-based constraint."""
        with patch(
            "pysensors.utils._constraints.load_functional_constraints"
        ) as mock_load:
            mock_load.return_value = lambda x, y: x**2 + y**2 - 4
            constraint = UserDefinedConstraints(
                all_sensors=all_sensors,
                file="test_constraint.py",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            mock_ax = MagicMock()
            mock_ax.scatter = MagicMock()
            with patch.object(
                BaseConstraint, "functional_constraints"
            ) as mock_func_constr:
                mock_func_constr.return_value = np.array([1, -1, 0, 2, -2])
                with patch.object(
                    BaseConstraint, "get_functionalConstraind_sensors_indices"
                ) as mock_get_func:
                    mock_get_func.return_value = (
                        [0, 2],
                        [0, 1],
                    )
                    with patch(
                        "pysensors.utils._constraints.get_coordinates_from_indices"
                    ) as mock_get_coords:
                        mock_get_coords.return_value = (
                            np.array([1, 3]),
                            np.array([5, 3]),
                        )
                        constraint.draw(mock_ax)
                        assert mock_func_constr.called
                        assert mock_get_func.called
                        assert mock_get_coords.called
                        assert mock_ax.scatter.called
                        call_args = mock_ax.scatter.call_args
                        args, kwargs = call_args
                        assert len(args) == 2
                        assert args[0].shape == (2,)
                        assert args[1].shape == (2,)
                        assert kwargs["s"] == 1

    def test_constraint_with_equation(self, all_sensors, sample_dataframe):
        """Test the constraint method with equation-based constraint."""
        constraint = UserDefinedConstraints(
            all_sensors=all_sensors,
            equation="x**2 + y**2 <= 4",
            data=sample_dataframe,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords:
            mock_get_coords.return_value = (np.array([1, 2, 3]), np.array([1, 2, 3]))
            with patch("builtins.eval") as mock_eval:
                mock_eval.side_effect = [
                    True,
                    False,
                    True,
                ]
                with patch.object(
                    BaseConstraint, "get_functionalConstraind_sensors_indices"
                ) as mock_get_func:
                    mock_get_func.return_value = (
                        [0, 2],
                        [0, 1],
                    )
                    idx_const, rank = constraint.constraint()
                    assert idx_const == [0, 2]
                    assert rank == [0, 1]
                    assert mock_get_coords.called
                    assert mock_eval.call_count == 3
                    assert mock_get_func.called

    def test_constraint_with_file(self, all_sensors, sample_dataframe):
        """Test the constraint method with file-based constraint."""
        with patch(
            "pysensors.utils._constraints.load_functional_constraints"
        ) as mock_load:
            mock_load.return_value = lambda x, y: x**2 + y**2 - 4
            constraint = UserDefinedConstraints(
                all_sensors=all_sensors,
                file="test_constraint.py",
                data=sample_dataframe,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            )
            with patch.object(
                BaseConstraint, "functional_constraints"
            ) as mock_func_constr:
                mock_func_constr.return_value = np.array([1, -1, 0, 2, -2])
                with patch.object(
                    BaseConstraint, "get_functionalConstraind_sensors_indices"
                ) as mock_get_func:
                    mock_get_func.return_value = (
                        [1, 2, 4],
                        [0, 1, 2],
                    )
                    idx_const, rank = constraint.constraint()
                    assert idx_const == [1, 2, 4]
                    assert rank == [0, 1, 2]
                    assert mock_func_constr.called
                    assert mock_get_func.called

    def test_with_real_equation(self, all_sensors):
        """Test with a real equation (no mocking)."""
        x = np.linspace(-3, 3, 7)
        y = np.linspace(-3, 3, 7)
        X, Y = np.meshgrid(x, y)

        df = pd.DataFrame(
            {
                "x_coord": X.flatten(),
                "y_coord": Y.flatten(),
                "field_val": np.random.random(49),
            }
        )
        sensors = np.arange(49)
        pytest.skip("This test requires the real implementation of the methods")
        constraint = UserDefinedConstraints(
            all_sensors=sensors,
            equation="x**2 + y**2 <= 4",
            data=df,
            X_axis="x_coord",
            Y_axis="y_coord",
            Field="field_val",
        )
        idx_const, rank = constraint.constraint()
        for idx in idx_const:
            x = df.iloc[idx]["x_coord"]
            y = df.iloc[idx]["y_coord"]
            assert x**2 + y**2 <= 4 + 1e-10

    def test_with_real_plot(self, all_sensors):
        """Test with a real plot (visual inspection only)."""
        pytest.skip("This test creates a real plot and is for manual inspection only")
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        X, Y = np.meshgrid(x, y)

        df = pd.DataFrame(
            {
                "x_coord": X.flatten(),
                "y_coord": Y.flatten(),
                "field_val": np.random.random(400),
            }
        )
        sensors = np.arange(400)
        constraints = [
            UserDefinedConstraints(
                all_sensors=sensors,
                equation="x**2 + y**2 <= 4",
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            UserDefinedConstraints(
                all_sensors=sensors,
                equation="abs(x) + abs(y) <= 2",
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
            UserDefinedConstraints(
                all_sensors=sensors,
                equation="y >= x**2 - 2",
                data=df,
                X_axis="x_coord",
                Y_axis="y_coord",
                Field="field_val",
            ),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax in axes:
            scatter = ax.scatter(
                df["x_coord"],
                df["y_coord"],
                c=df["field_val"],
                cmap="coolwarm",
                alpha=0.3,
                s=10,
            )
        for i, (ax, constraint) in enumerate(zip(axes, constraints)):
            constraint.draw(ax)
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_title(["Circle", "Diamond", "Parabola"][i])
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label="Field Value")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()

    def test_draw_with_numpy_array_and_equation(self):
        """Test draw method with numpy array data and equation-based constraints."""
        sample_array = np.random.rand(5, 16)
        all_sensors = np.array([0, 1, 2, 3, 4])
        constraint = UserDefinedConstraints(
            all_sensors=all_sensors, equation="x**2 + y**2 <= 4", data=sample_array
        )
        with patch(
            "pysensors.utils._constraints.get_coordinates_from_indices"
        ) as mock_get_coords, patch("builtins.eval") as mock_eval, patch.object(
            BaseConstraint, "get_functionalConstraind_sensors_indices"
        ) as mock_get_func:
            x_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            y_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mock_get_coords.side_effect = [
                (x_values, y_values),
                (x_values[:2], y_values[:2]),
            ]
            mock_eval.side_effect = [False, False, True, True, True]
            mock_get_func.return_value = (np.array([0, 1]), np.array([0, 1]))
            mock_ax = MagicMock()

            constraint.draw(mock_ax)

            assert mock_get_coords.call_count == 2
            args1, kwargs1 = mock_get_coords.call_args_list[0]
            np.testing.assert_array_equal(args1[0], all_sensors)
            np.testing.assert_array_equal(args1[1], sample_array)

            assert mock_eval.call_count == 5
            args2, kwargs2 = mock_get_func.call_args
            np.testing.assert_array_equal(args2[0], all_sensors)
            expected_g = np.array([True, True, False, False, False])
            np.testing.assert_array_equal(args2[1], expected_g)

            args3, kwargs3 = mock_get_coords.call_args_list[1]
            np.testing.assert_array_equal(args3[0], np.array([0, 1]))
            np.testing.assert_array_equal(args3[1], sample_array)

            mock_ax.scatter.assert_called_once()
            args4, kwargs4 = mock_ax.scatter.call_args
            np.testing.assert_array_equal(args4[0], x_values[:2])
            np.testing.assert_array_equal(args4[1], y_values[:2])
            assert kwargs4["s"] == 1

    def test_draw_with_file_and_numpy_array(self):
        """Test the draw method with file-based constraint and numpy array data."""
        sample_array = np.random.rand(5, 16)
        all_sensors = np.array([0, 1, 2, 3, 4])
        with patch(
            "pysensors.utils._constraints.load_functional_constraints"
        ) as mock_load:

            def test_function(x, y):
                return x**2 + y**2 - 4

            mock_load.return_value = test_function
            constraint = UserDefinedConstraints(
                all_sensors=all_sensors, file="test_constraint.py", data=sample_array
            )
            assert constraint.functions == test_function
            with patch.object(
                BaseConstraint, "functional_constraints"
            ) as mock_func_constr, patch.object(
                BaseConstraint, "get_functionalConstraind_sensors_indices"
            ) as mock_get_func, patch(
                "pysensors.utils._constraints.get_coordinates_from_indices"
            ) as mock_get_coords:
                mock_func_constr.return_value = np.array([1.5, -0.5, 0.8, -1.2, 2.3])
                mock_get_func.return_value = (np.array([0, 2, 4]), np.array([0, 1, 2]))
                mock_get_coords.return_value = (
                    np.array([1, 3, 5]),
                    np.array([2, 4, 6]),
                )
                mock_ax = MagicMock()
                constraint.draw(mock_ax)
                mock_func_constr.assert_called_once()
                func_args, func_kwargs = mock_func_constr.call_args
                assert func_args[0] == test_function
                np.testing.assert_array_equal(func_args[1], all_sensors)
                assert func_args[2] is sample_array
                expected_g = np.array([True, False, True, False, True])
                mock_get_func.assert_called_once()
                get_func_args = mock_get_func.call_args[0]
                np.testing.assert_array_equal(get_func_args[0], all_sensors)
                np.testing.assert_array_equal(get_func_args[1], expected_g)
                mock_get_coords.assert_called_once()
                get_coords_args = mock_get_coords.call_args[0]
                np.testing.assert_array_equal(get_coords_args[0], np.array([0, 2, 4]))
                assert get_coords_args[1] is sample_array
                mock_ax.scatter.assert_called_once()
                scatter_args = mock_ax.scatter.call_args[0]
                np.testing.assert_array_equal(scatter_args[0], np.array([1, 3, 5]))
                np.testing.assert_array_equal(scatter_args[1], np.array([2, 4, 6]))
                scatter_kwargs = mock_ax.scatter.call_args[1]
                assert scatter_kwargs["s"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
