# TODO: include some unit tests once there are more functions
# in this submodule
import numpy as np
import pytest
import pandas as pd
from pysensors.utils._constraints import get_constrained_sensors_indices
from pysensors.utils._constraints import get_constrained_sensors_indices_dataframe
from pysensors.utils._constraints import load_functional_constraints

## Testing get_constrained_sensors_indices
def test_get_constrained_sensors_indices_empty_array():
    all_sensors = np.array([])
    x_min, x_max, y_min, y_max, nx, ny = 0, 10, 0, 10, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_get_constrained_sensors_indices_non_integer_values():
    all_sensors = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    x_min, x_max, y_min, y_max, nx, ny = 2, 4, 3, 5, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_get_constrained_sensors_indices_no_constrained_sensors():
    all_sensors = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    x_min, x_max, y_min, y_max, nx, ny = 6, 8, 9, 11, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_get_constrained_sensors_indices_single_constrained_sensor():
    x_min, x_max, y_min, y_max, nx, ny = 3, 4, 3, 4, 10, 10
    all_sensors = np.array([i+101 for i in range(nx*ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_get_constrained_sensors_indices_multiple_constrained_sensors():
    all_sensors = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5], [9.5, 10.5]])
    x_min, x_max, y_min, y_max, nx, ny = 3, 7, 3, 7, 10, 10
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_valid_input_parameters():
    nx, ny, x_min, x_max, y_min, y_max  = 10, 10, 2, 8, 2, 8
    all_sensors = np.array([i for i in range(nx*ny)])
    result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)
    assert len(result) == (x_max - x_min + 1) * (y_max - y_min + 1)

def test_one_constrained_sensor():
    nx, ny, x_min, x_max, y_min, y_max  = 10, 10, 8, 9, 8, 9
    all_sensors = np.array([i for i in range(nx*ny)])
    result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)
    assert len(result) == 4

def test_invalid_nx_not_integer():
    nx, ny, x_min, x_max, y_min, y_max = 'ten', 10, 2, 8, 2, 8
    all_sensors = np.array([i for i in range(ny**2)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_invalid_ny_not_integer():
    nx, ny, x_min, x_max, y_min, y_max = 10, 'ten', 2, 8, 2, 8
    all_sensors = np.array([i for i in range(nx**2)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_invalid_x_min_greater_than_x_max():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 8, 2, 2, 8
    all_sensors = np.array([i for i in range(nx*ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_invalid_y_min_greater_than_y_max():
    nx, ny, x_min, x_max, y_min, y_max = 10, 10, 2, 8, 8, 2
    all_sensors = np.array([i for i in range(nx*ny)])
    with pytest.raises(ValueError):
        result = get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors)

def test_get_constrained_sensors_indices_dataframe_does_not_modify_input_dataframe():
    seed = 8051977
    # Create a test dataframe
    test_dataframe = pd.DataFrame({
        'x': np.random.randint(0, 100, size=100),
        'y': np.random.randint(0, 100, size=100),
        'Field': np.random.randint(0, 100, size=100)
    })
    df = test_dataframe.copy()
    # Define test parameters
    x_min, x_max, y_min, y_max = 50, 75, 25, 50

    # Call the function
    idx_constrained = get_constrained_sensors_indices_dataframe(x_min, x_max, y_min, y_max, test_dataframe, X_axis='x', Y_axis='y')

    # Assert that the input dataframe is not modified
    assert test_dataframe.equals(df)





## Test load_functional_constraints
def test_load_functional_constraints_loads_valid_python_file():
    """
    Test that the function loads a valid Python file and returns a callable function.
    """
    test_file = "test_function.py"
    with open(test_file, "w") as f:
        f.write("""
def test_function():
    return 1
    """)
    func = load_functional_constraints(test_file)
    assert func.__name__ == "test_function"
    assert func() == 1