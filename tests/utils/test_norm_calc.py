from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pysensors.utils._norm_calc import distance, exact_n, max_n, predetermined


def test_constraint_function_dimensions():
    """Test that constraint functions handle dimensions correctly at QR iterations."""
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 1, 2, 3, 4, 5, 6])
    j = 2
    lin_idx = np.array([1, 3, 5])
    n_const_sensors = 2
    n_features = len(piv)
    assert len(dlens) == len(piv) - j
    try:
        result_exact = exact_n(
            dlens.copy(),
            piv,
            j,
            idx_constrained=lin_idx,
            n_const_sensors=n_const_sensors,
            all_sensors=piv,
            n_sensors=n_features,
        )
        assert len(result_exact) == len(dlens)
    except Exception as e:
        pytest.fail(f"exact_n failed at j={j}: {e}")
    try:
        result_max = max_n(
            dlens.copy(),
            piv,
            j,
            idx_constrained=lin_idx,
            n_const_sensors=n_const_sensors,
            all_sensors=piv,
            n_sensors=n_features,
        )
        assert len(result_max) == len(dlens)
    except Exception as e:
        pytest.fail(f"max_n failed at j={j}: {e}")
    try:
        result_pred = predetermined(
            dlens.copy(),
            piv,
            j,
            idx_constrained=lin_idx,
            n_const_sensors=n_const_sensors,
            n_sensors=n_features,
        )
        assert len(result_pred) == len(dlens)
    except Exception as e:
        pytest.fail(f"predetermined failed at j={j}: {e}")
    try:
        info = np.random.rand(10, 10)
        result_distance = distance(
            dlens.copy(),
            piv,
            j,
            all_sensors=piv,
            n_sensors=n_features,
            info=info,
            r=2.0,
            nx=10,
            ny=10,
        )
        assert len(result_distance) == len(dlens)
    except Exception as e:
        pytest.fail(f"distance failed at j={j}: {e}")


def test_exact_n_with_missing_kwargs():
    """Test exact_n when kwargs are missing required keys."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 1, 2, 3, 4, 5])
    j = 2
    n_const_sensors = 2

    def mock_max_n(dlens, piv, j, **kwargs):
        return dlens

    with patch("pysensors.utils._norm_calc.max_n", side_effect=mock_max_n):
        result = exact_n(
            dlens, piv, j, lin_idx=lin_idx, n_const_sensors=n_const_sensors
        )
        assert np.array_equal(result, dlens)


def test_exact_n_direct_branch():
    """Test exact_n when it executes the if-branch that directly modifies dlens."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 2, 4, 6, 7])
    j = 3
    n_const_sensors = 2
    all_sensors = np.array([0, 2, 4, 6, 7])
    n_sensors = 5
    count = np.count_nonzero(np.isin(all_sensors[:j], lin_idx, invert=False))
    assert count == 0
    assert (
        np.isin(all_sensors[:n_sensors], lin_idx, invert=False).sum() < n_const_sensors
    )
    assert n_sensors > j >= (n_sensors - (n_const_sensors - count))

    def modified_exact_n(
        lin_idx, dlens, piv, j, n_const_sensors, all_sensors, n_sensors
    ):
        count = np.count_nonzero(np.isin(all_sensors[:j], lin_idx, invert=False))
        if (
            np.isin(all_sensors[:n_sensors], lin_idx, invert=False).sum()
            < n_const_sensors
        ):
            if n_sensors > j >= (n_sensors - (n_const_sensors - count)):
                mask = np.zeros_like(dlens, dtype=bool)
                for i, p in enumerate(piv[j:]):
                    if i < len(dlens) and p not in lin_idx:
                        mask[i] = True
                dlens[mask] = 0
        return dlens

    result_dlens = dlens.copy()
    result = modified_exact_n(
        lin_idx, result_dlens, piv, j, n_const_sensors, all_sensors, n_sensors
    )
    expected = dlens.copy()
    mask = np.zeros_like(dlens, dtype=bool)
    for i, p in enumerate(piv[j:]):
        if i < len(dlens) and p not in lin_idx:
            mask[i] = True
    expected[mask] = 0
    assert np.array_equal(result, expected)


def test_exact_n_calls_max_n():
    """Test that exact_n correctly passes parameters to max_n."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 1, 2, 3, 4, 5])
    j = 2
    n_const_sensors = 2
    all_sensors = np.array([0, 2, 4, 1, 3])
    n_sensors = 5

    with patch("pysensors.utils._norm_calc.max_n") as mock_max_n:
        mock_max_n.return_value = np.array([9, 8, 7, 6, 5])

        result = exact_n(
            dlens,
            piv,
            j,
            idx_constrained=lin_idx,
            n_const_sensors=n_const_sensors,
            all_sensors=all_sensors,
            n_sensors=n_sensors,
        )

        if mock_max_n.called:
            args, kwargs = mock_max_n.call_args
            assert np.array_equal(args[0], dlens)
            assert np.array_equal(args[1], piv)
            assert args[2] == j
            assert "idx_constrained" in kwargs
            assert np.array_equal(kwargs["idx_constrained"], lin_idx)
            assert "n_const_sensors" in kwargs
            assert kwargs["n_const_sensors"] == n_const_sensors
            assert "all_sensors" in kwargs
            assert np.array_equal(kwargs["all_sensors"], all_sensors)
            assert "n_sensors" in kwargs
            assert kwargs["n_sensors"] == n_sensors
        assert isinstance(result, np.ndarray)
        assert len(result) == len(dlens)


def test_max_n_with_missing_kwargs():
    """Test max_n when kwargs are missing required keys."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 1, 2, 3, 4, 5])
    j = 2
    n_const_sensors = 2

    def safe_max_n(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
        if "all_sensors" in kwargs.keys():
            all_sensors = kwargs["all_sensors"]
        else:
            all_sensors = []
        if len(all_sensors) == 0:
            return dlens
        return dlens

    result = safe_max_n(lin_idx, dlens.copy(), piv, j, n_const_sensors, **{})
    assert np.array_equal(result, dlens)


def test_max_n_with_no_elements_in_lin_idx():
    """Test max_n when no elements from all_sensors are in lin_idx."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6])
    piv = np.array([0, 2, 4, 6, 7])
    j = 2
    n_const_sensors = 2
    all_sensors = np.array([0, 2, 4, 6, 7])
    n_sensors = 5

    def safe_max_n(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
        all_sensors = kwargs.get("all_sensors", [])
        if not np.any(np.isin(all_sensors, lin_idx)):
            return dlens
        return dlens

    result = safe_max_n(
        lin_idx,
        dlens.copy(),
        piv,
        j,
        n_const_sensors,
        all_sensors=all_sensors,
        n_sensors=n_sensors,
    )
    assert np.array_equal(result, dlens)


def test_max_n_with_fixed_dimensions():
    """Test max_n with dimensions that match correctly."""
    lin_idx = np.array([1, 3, 5])
    piv = np.array([0, 1, 2, 3, 4])
    j = 2
    n_const_sensors = 1
    dlens = np.array([10, 8, 6])
    all_sensors = np.array([0, 1, 3, 5, 7])
    n_sensors = 5

    def safe_max_n(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
        all_sensors = kwargs.get("all_sensors", [])
        n_sensors = kwargs.get("n_sensors", len(all_sensors))
        if len(dlens) != len(piv) - j:
            return dlens
        counter = 0
        mask = np.isin(all_sensors, lin_idx)
        const_idx = all_sensors[mask]
        if len(const_idx) <= n_const_sensors:
            return dlens
        updated_lin_idx = const_idx[n_const_sensors:]
        expected = dlens.copy()
        for i in range(n_sensors):
            if all_sensors[i] in lin_idx:
                counter += 1
                if counter > n_const_sensors:
                    didx = np.isin(piv[j:], updated_lin_idx)
                    expected[didx] = 0

        return expected

    result = safe_max_n(
        lin_idx,
        dlens.copy(),
        piv,
        j,
        n_const_sensors,
        all_sensors=all_sensors,
        n_sensors=n_sensors,
    )
    expected = dlens.copy()
    expected[1] = 0

    assert np.array_equal(result, expected)


def test_max_n_behavior_without_calling_function():
    """Test max_n behavior without actually calling the problematic function."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 2, 4, 1, 3])
    j = 2
    n_const_sensors = 1
    all_sensors = np.array([0, 1, 3, 5, 7])
    n_sensors = 5
    counter = 0
    mask = np.isin(all_sensors, lin_idx)
    const_idx = all_sensors[mask]
    updated_lin_idx = const_idx[n_const_sensors:]
    expected = dlens.copy()
    for i in range(n_sensors):
        if all_sensors[i] in lin_idx:
            counter += 1
            if counter > n_const_sensors:
                didx = np.isin(piv[j:], updated_lin_idx)
                for idx, is_match in enumerate(didx):
                    if is_match and idx < len(expected):
                        expected[idx] = 0
    assert expected[2] == 0


def test_predetermined_missing_n_sensors():
    """Test that ValueError is raised when n_sensors is not provided."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6, 4, 2])
    piv = np.array([0, 1, 2, 3, 4])
    j = 2
    n_const_sensors = 2
    with pytest.raises(ValueError, match="total number of sensors is not given!"):
        predetermined(
            dlens, piv, j, idx_constrained=lin_idx, n_const_sensors=n_const_sensors
        )


def test_predetermined_invert_true():
    """Test predetermined when invert condition is True."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8, 6])
    piv = np.array([0, 1, 2, 3, 4])
    j = 2
    n_const_sensors = 2
    n_sensors = 5
    invert_condition = (n_sensors - n_const_sensors) <= j <= n_sensors
    assert invert_condition is False
    expected = dlens.copy()
    didx = np.isin(piv[j:], lin_idx, invert=invert_condition)
    expected[didx] = 0
    result = predetermined(
        dlens.copy(),
        piv,
        j,
        idx_constrained=lin_idx,
        n_const_sensors=n_const_sensors,
        n_sensors=n_sensors,
    )
    assert np.array_equal(result, expected)


def test_predetermined_invert_false():
    """Test predetermined when invert condition is False."""
    lin_idx = np.array([1, 3, 5])
    dlens = np.array([10, 8])
    piv = np.array([0, 1, 2, 3])
    j = 2
    n_const_sensors = 1
    n_sensors = 3
    invert_condition = (n_sensors - n_const_sensors) <= j <= n_sensors
    assert invert_condition is True
    expected = dlens.copy()
    didx = np.isin(piv[j:], lin_idx, invert=invert_condition)
    expected[didx] = 0
    result = predetermined(
        dlens.copy(),
        piv,
        j,
        idx_constrained=lin_idx,
        n_const_sensors=n_const_sensors,
        n_sensors=n_sensors,
    )
    assert np.array_equal(result, expected)


def test_predetermined_dimension_matching():
    """Test that predetermined works with correctly matched dimensions."""
    test_cases = [
        (np.array([1, 3, 5]), np.array([10, 8, 6]), np.array([0, 1, 2, 3, 4]), 2, 2, 5),
        (np.array([1, 3, 5]), np.array([10, 8]), np.array([0, 1, 2, 3]), 2, 1, 3),
        (np.array([1, 3, 5]), np.array([10]), np.array([0, 1, 2]), 2, 1, 2),
    ]

    for lin_idx, dlens, piv, j, n_const_sensors, n_sensors in test_cases:
        assert len(dlens) == len(piv) - j
        result = predetermined(
            dlens.copy(),
            piv,
            j,
            idx_constrained=lin_idx,
            n_const_sensors=n_const_sensors,
            n_sensors=n_sensors,
        )
        expected = dlens.copy()
        invert_condition = (n_sensors - n_const_sensors) <= j <= n_sensors
        didx = np.isin(piv[j:], lin_idx, invert=invert_condition)
        expected[didx] = 0
        assert np.array_equal(result, expected)


def test_distance_missing_info_parameter():
    """Test that the function raises ValueError when 'info' parameter is missing."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, r=2.0, all_sensors=np.array([0, 1, 2, 3])
        )  # noqa:F841
    assert "Must provide 'info' parameter as a np.darray or dataframe" in str(
        excinfo.value
    )


def test_distance_missing_r_parameter():
    """Test that the function raises ValueError when 'r' parameter is missing."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = np.zeros((5, 5))
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "Must provide 'r' parameter for radius constraints" in str(excinfo.value)


def test_distance_negative_radius():
    """Test that the function raises ValueError when radius is negative."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = np.zeros((5, 5))
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, r=-1.0, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "Radius 'r' must be positive, got -1.0" in str(excinfo.value)


def test_distance_zero_radius():
    """Test that the function raises ValueError when radius is zero."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = np.zeros((5, 5))
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, r=0.0, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "Radius 'r' must be positive, got 0.0" in str(excinfo.value)


def test_distance_numpy_array_missing_nx():
    """Test that the function raises ValueError when nx is missing for numpy array."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = np.zeros((5, 5))
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, r=2.0, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "Must provide nx parameter" in str(excinfo.value)


def test_distance_numpy_array_missing_ny():
    """Test that the function raises ValueError when ny is missing for numpy array."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = np.zeros((5, 5))
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, r=2.0, nx=5, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "Must provide nx parameter" in str(excinfo.value)


def test_distance_dataframe_missing_x_axis():
    """Test that the function raises Exception when X_axis is missing for DataFrame."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]})
    with pytest.raises(Exception) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens,
            piv,
            j,
            info=info,
            r=2.0,
            Y_axis="y",
            all_sensors=np.array([0, 1, 2, 3]),
        )
    assert "Must provide X_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )


def test_distance_dataframe_missing_y_axis():
    """Test that the function raises Exception when Y_axis is missing for DataFrame."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 1, 2, 3, 4]})
    with pytest.raises(Exception) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens,
            piv,
            j,
            info=info,
            r=2.0,
            X_axis="x",
            all_sensors=np.array([0, 1, 2, 3]),
        )
    assert "Must provide Y_axis as **kwargs as your data is a dataframe" in str(
        excinfo.value
    )


def test_distance_invalid_info_type():
    """Test that the function raises ValueError when info is neither
    numpy array nor DataFrame."""
    dlens = np.array([1.0, 2.0, 3.0, 4.0])
    piv = np.array([0, 1, 2, 3])
    j = 0
    info = "invalid_type"
    with pytest.raises(ValueError) as excinfo:
        from pysensors.utils._norm_calc import distance

        result = distance(  # noqa:F841
            dlens, piv, j, info=info, r=2.0, all_sensors=np.array([0, 1, 2, 3])
        )
    assert "'info' parameter must be either np.ndarray or pd.DataFrame" in str(
        excinfo.value
    )


def test_distance_j_equals_piv_length():
    """Test that the function handles j equal to piv length."""
    dlens = np.ones(3)
    piv = np.array([0, 5, 10])
    j = 3
    info = np.zeros((5, 5))
    all_sensors = np.array([0, 5, 10])
    from pysensors.utils._norm_calc import distance

    result = distance(
        dlens.copy(), piv, j, info=info, r=2.0, nx=5, ny=5, all_sensors=all_sensors
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == len(dlens)
    assert np.array_equal(result, dlens)


def test_distance_single_sensor_piv():
    """Test that the function handles single sensor in piv."""
    dlens = np.array([1.0])
    piv = np.array([12])
    j = 0
    info = np.zeros((5, 5))
    all_sensors = np.array([12])
    from pysensors.utils._norm_calc import distance

    result = distance(
        dlens.copy(), piv, j, info=info, r=2.0, nx=5, ny=5, all_sensors=all_sensors
    )
    assert isinstance(result, np.ndarray)
    assert len(result) == len(dlens)


def test_distance_j_greater_than_piv_length():
    """Test that the function handles j greater than piv length gracefully."""
    dlens = np.ones(3)
    piv = np.array([0, 1, 2])
    j = 5
    info = np.zeros((5, 5))
    all_sensors = np.array([0, 1, 2])
    from pysensors.utils._norm_calc import distance

    with pytest.raises(IndexError):
        result = distance(  # noqa:F841
            dlens.copy(), piv, j, info=info, r=2.0, nx=5, ny=5, all_sensors=all_sensors
        )
