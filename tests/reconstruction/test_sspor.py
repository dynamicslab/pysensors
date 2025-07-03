"""
Unit tests for SSPOR class (and any other classes implemented
in pysensors.py).

Note: all tests should be encapsulated in functions whose
names start with "test_"

To run all tests for this package, navigate to the top-level
directory and execute the following command:
pytest

To run tests for just one file, run
pytest file_to_test.py
"""

import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy import isnan, mean, nan, sqrt, zeros
from pytest_lazyfixture import lazy_fixture
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import SVD, Identity, RandomProjection
from pysensors.optimizers import CCQR, TPGR
from pysensors.reconstruction import SSPOR


def test_not_fitted(data_vandermonde):
    x = data_vandermonde
    model = SSPOR()
    prior = np.full(2, 1)

    # Should not be able to call any of these methods before fitting
    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.get_selected_sensors()
    with pytest.raises(NotFittedError):
        model.get_all_sensors()
    with pytest.raises(NotFittedError):
        model.set_number_of_sensors(20)
    with pytest.raises(NotFittedError):
        model.score(x)
    with pytest.raises(NotFittedError):
        model.reconstruction_error(x)
    with pytest.raises(NotFittedError):
        model.std(prior)


def test_set_number_of_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    model = SSPOR()
    model.fit(x)

    with pytest.raises(ValueError):
        model.set_number_of_sensors(max_sensors + 1)
    with pytest.raises(ValueError):
        model.set_number_of_sensors(0)
    with pytest.raises(ValueError):
        model.set_number_of_sensors(1.5)
    with pytest.raises(ValueError):
        model.set_number_of_sensors("3")

    model.set_number_of_sensors(15)
    assert len(model.get_selected_sensors()) == 15


@pytest.mark.parametrize(
    "data",
    [lazy_fixture("data_vandermonde"), lazy_fixture("data_random")],
)
def test_get_all_sensors(data):
    x = data
    max_sensors = x.shape[1]

    model = SSPOR()
    model.fit(x)
    assert len(model.get_all_sensors()) == max_sensors


@pytest.mark.parametrize(
    "basis", [Identity(), SVD(), RandomProjection(n_basis_modes=5)]
)
def test_basis_compatibility(data_vandermonde, basis):
    x = data_vandermonde
    model = SSPOR(basis=basis)
    model.fit(x)
    check_is_fitted(model)


def test_n_sensors(data_random):

    # Check for bad inputs
    with pytest.raises(ValueError):
        model = SSPOR(n_sensors=0)
    with pytest.raises(ValueError):
        model = SSPOR(n_sensors=5.4)
    with pytest.raises(ValueError):
        model = SSPOR(n_sensors="1")
    with pytest.raises(ValueError):
        model = SSPOR(n_sensors=[1])

    n_sensors = 5
    x = data_random
    model = SSPOR(n_sensors=n_sensors)
    model.fit(x)

    assert len(model.get_selected_sensors()) == n_sensors


def test_predict(data_random):
    data = data_random

    n_sensors = 5
    model = SSPOR(n_sensors=n_sensors)
    model.fit(data)

    # Wrong size input for predict
    # (should only pass data at sensor locations)
    with pytest.raises(ValueError):
        model.predict(data)

    # Rectangular case
    sensors = model.get_selected_sensors()
    assert data.shape == model.predict(data[:, sensors]).shape


def test_square_predict(data_random_square):
    data = data_random_square

    model = SSPOR()
    model.fit(data)
    sensors = model.get_selected_sensors()
    assert data.shape == model.predict(data[:, sensors]).shape


def test_predict_accuracy(data_vandermonde_testing):
    # Polynomials up to degree 10 on [0, 1]
    data, x_test = data_vandermonde_testing

    model = SSPOR()
    model.fit(data, seed=1)
    model.set_number_of_sensors(8)
    sensors = model.get_selected_sensors()
    assert (
        sqrt(
            mean((x_test - model.predict(x_test[sensors], method="unregularized")) ** 2)
        )
        <= 1.0e-3
    )

    # Should also work for row vectors
    x_test = x_test.reshape(1, -1)
    assert (
        sqrt(
            mean(
                (x_test - model.predict(x_test[:, sensors], method="unregularized"))
                ** 2
            )
        )
        <= 1.0e-3
    )


def test_reconstruction_error(data_vandermonde_testing):
    data, x_test = data_vandermonde_testing

    model = SSPOR(n_sensors=3)
    model.fit(data)

    assert len(model.reconstruction_error(x_test)) == min(
        model.n_sensors, data.shape[1]
    )

    sensor_range = [1, 2, 3]
    assert len(model.reconstruction_error(x_test, sensor_range=sensor_range)) == 3


def test_score(data_vandermonde):
    data = data_vandermonde

    weak_model = SSPOR(n_sensors=3)
    weak_model.fit(data)

    # You must pass in data with as many features as the training set
    with pytest.raises(ValueError):
        weak_model.score(data[:, :5])

    strong_model = SSPOR(n_sensors=8)
    strong_model.fit(data)

    assert weak_model.score(data) < strong_model.score(data)


def test_prefit_basis(data_random):
    data = data_random
    basis = Identity()
    basis.fit(data)

    # This data should be ignored during the fit
    data_to_ignore = nan * data_random

    model = SSPOR(basis=basis)
    model.fit(data_to_ignore, prefit_basis=True)
    assert not any(isnan(model.get_selected_sensors()))


def test_update_n_basis_modes_errors(data_random):
    data = data_random
    n_basis_modes = 5
    model = SSPOR(basis=Identity(n_basis_modes=n_basis_modes))

    model.fit(data)

    with pytest.raises(ValueError):
        model.update_n_basis_modes(0)
    with pytest.raises(ValueError):
        model.update_n_basis_modes("5")
    with pytest.raises(ValueError):
        model.update_n_basis_modes(data.shape[0] + 1)
    # Need to pass x when increasing n_basis_modes beyond capacity
    # of the original basis
    with pytest.raises(ValueError):
        model.update_n_basis_modes(n_basis_modes + 1)


def test_update_n_basis_modes(data_random):
    data = data_random
    model = SSPOR()
    model.fit(data)
    assert model.basis.n_basis_modes == data.shape[0]
    assert model.basis_matrix_.shape[1] == data.shape[0]

    n_basis_modes = 5
    model.update_n_basis_modes(n_basis_modes, x=data)
    assert model.basis.n_basis_modes == data.shape[0]
    assert model.basis_matrix_.shape[1] == n_basis_modes


def test_update_n_basis_modes_refit(data_random):
    data = data_random
    n_basis_modes = 5
    model = SSPOR(basis=Identity(n_basis_modes=n_basis_modes))
    model.fit(data)
    assert model.basis_matrix_.shape[1] == n_basis_modes

    model.update_n_basis_modes(n_basis_modes + 1, data)
    assert model.basis_matrix_.shape[1] == n_basis_modes + 1


def test_update_n_basis_modes_unfit_basis(data_random):
    data = data_random
    n_basis_modes = 5
    model = SSPOR()
    model.update_n_basis_modes(n_basis_modes, data)

    assert model.basis_matrix_.shape[1] == n_basis_modes


def test_ccqr_integration(data_random):
    data = data_random
    costs = zeros(data.shape[1])
    costs[[1, 3, 5]] = 100

    optimizer = CCQR(sensor_costs=costs)
    model = SSPOR(optimizer=optimizer).fit(data)

    check_is_fitted(model)


def test_sensor_selector_properties(data_random):
    data = data_random
    model = SSPOR().fit(data)

    assert all(model.get_all_sensors() == model.all_sensors)
    assert all(model.get_selected_sensors() == model.selected_sensors)


class MockBasisWithWarning(BaseEstimator):
    """Mock basis class that raises warnings when fit is called."""

    def __init__(self, n_basis_modes=3):
        self.n_basis_modes = n_basis_modes

    def fit(self, x):
        warnings.warn("This is a test warning from basis fit", UserWarning)
        self.basis_matrix_ = np.random.rand(x.shape[1], self.n_basis_modes)
        return self

    def matrix_representation(self, n_basis_modes=None):
        return self.basis_matrix_


@pytest.fixture
def sspor_with_warning_basis():
    """Create an SSPOR instance with a basis that raises warnings."""
    basis = MockBasisWithWarning()
    optimizer = Mock()
    optimizer.fit.return_value = optimizer
    optimizer.get_sensors.return_value = np.arange(5)
    sspor = SSPOR(basis=basis, optimizer=optimizer, n_sensors=3)

    return sspor


def test_warnings_shown_with_quiet_false(sspor_with_warning_basis):
    """Test that warnings are shown when quiet=False."""
    X = np.random.rand(10, 5)
    with pytest.warns(UserWarning, match="This is a test warning from basis fit"):
        sspor_with_warning_basis.fit(X, quiet=False)


def test_warnings_suppressed_with_quiet_true(sspor_with_warning_basis):
    """Test that warnings are suppressed when quiet=True."""
    X = np.random.rand(10, 5)
    with warnings.catch_warnings(record=True) as recorded_warnings:
        sspor_with_warning_basis.fit(X, quiet=True)
        assert len(recorded_warnings) == 0


def test_prefit_basis_skips_basis_fit(sspor_with_warning_basis):
    """Test that when prefit_basis=True, basis.fit is not called."""
    X = np.random.rand(10, 5)
    sspor_with_warning_basis.basis.basis_matrix_ = np.random.rand(5, 3)
    original_fit = sspor_with_warning_basis.basis.fit
    call_count = [0]

    def spy_fit(x):
        call_count[0] += 1
        return original_fit(x)

    sspor_with_warning_basis.basis.fit = spy_fit
    with warnings.catch_warnings(record=True) as recorded_warnings:
        with patch("sklearn.utils.validation.check_is_fitted"):
            sspor_with_warning_basis.fit(X, prefit_basis=True)
        assert len(recorded_warnings) == 0
    assert call_count[0] == 0


def test_predict_warns_when_n_sensors_exceeds_basis_dimension():
    """Test that warning is raised when n_sensors exceeds dimension of basis modes."""
    basis = Mock()
    optimizer = Mock()
    predictor = SSPOR(basis=basis, optimizer=optimizer)

    predictor.basis_matrix_ = np.random.rand(3, 2)
    predictor.n_sensors = 4
    predictor.ranked_sensors_ = np.array([0, 1, 2, 3, 4])
    predictor._rectangular_predict = Mock(return_value=np.array([[1, 2, 3]]))
    predictor._square_predict = Mock(return_value=np.array([[1, 2, 3]]))
    original_validate_input = predictor.predict.__globals__.get(
        "validate_input", lambda x, y: x.T
    )
    try:
        setattr(predictor, "check_is_fitted_", lambda: None)
        X = np.random.rand(2, 4)
        with pytest.warns(
            UserWarning, match="n_sensors exceeds dimension of basis modes"
        ):
            predictor.predict.__globals__["validate_input"] = lambda x, y: X.T
            predictor.predict(X, method="unregularized")
    finally:
        if "validate_input" in predictor.predict.__globals__:
            predictor.predict.__globals__["validate_input"] = original_validate_input


def test_predict_no_warning_when_n_sensors_not_exceeds_basis_dimension():
    """Test no warning when n_sensors does not exceed dimension of basis modes."""
    basis = Mock()
    optimizer = Mock()
    predictor = SSPOR(basis=basis, optimizer=optimizer)

    predictor.basis_matrix_ = np.random.rand(5, 2)
    predictor.n_sensors = 3
    predictor.ranked_sensors_ = np.array([0, 1, 2, 3, 4])

    predictor._rectangular_predict = Mock(return_value=np.array([[1, 2, 3, 4, 5]]))
    predictor._square_predict = Mock(return_value=np.array([[1, 2, 3, 4, 5]]))
    original_validate_input = predictor.predict.__globals__.get(
        "validate_input", lambda x, y: x.T
    )
    try:
        setattr(predictor, "check_is_fitted_", lambda: None)
        X = np.random.rand(2, 3)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            predictor.predict.__globals__["validate_input"] = lambda x, y: X.T
            predictor.predict(X, method="unregularized")
            relevant_warnings = [
                w
                for w in recorded_warnings
                if issubclass(w.category, UserWarning)
                and "n_sensors exceeds dimension of basis modes" in str(w.message)
            ]
            assert len(relevant_warnings) == 0
    finally:
        if "validate_input" in predictor.predict.__globals__:
            predictor.predict.__globals__["validate_input"] = original_validate_input


def test_reconstruction_error_warns_when_sensor_range_exceeds_basis_mode_dim():
    """Test that a warning is raised when sensor_range max exceeds basis_mode_dim."""
    basis = Mock()
    optimizer = Mock()
    sspor = SSPOR(basis=basis, optimizer=optimizer)
    basis_mode_dim = 3
    n_basis_modes = 2
    sspor.basis_matrix_ = np.random.rand(basis_mode_dim, n_basis_modes)
    sspor.ranked_sensors_ = np.array([0, 1, 2, 3, 4])
    sensor_range = np.array([1, 2, 3, 4])
    sspor.get_all_sensors = Mock(return_value=np.arange(5))
    sspor._square_predict = Mock(return_value=np.random.rand(5, 5))
    sspor._rectangular_predict = Mock(return_value=np.random.rand(5, 5))

    def run_test():
        if sensor_range[-1] > basis_mode_dim:
            warnings.warn(
                f"Performance may be poor when using more than {basis_mode_dim} sensors"
            )
        return np.zeros_like(sensor_range, dtype=np.float64)

    with pytest.warns(
        UserWarning,
        match=f"Performance may be poor when using more than {basis_mode_dim} sensors",
    ):
        run_test()


def test_reconstruction_error_no_warning_when_sensor_range_within_limit():
    """Test that no warning is raised when sensor_range max is within basis_mode_dim."""
    basis = Mock()
    optimizer = Mock()
    sspor = SSPOR(basis=basis, optimizer=optimizer)
    basis_mode_dim = 5
    n_basis_modes = 2
    sspor.basis_matrix_ = np.random.rand(basis_mode_dim, n_basis_modes)
    sspor.ranked_sensors_ = np.array([0, 1, 2, 3, 4])
    sensor_range = np.array([1, 2, 3])

    def run_test():
        if sensor_range[-1] > basis_mode_dim:
            warnings.warn(
                f"Performance may be poor when using more than {basis_mode_dim} sensors"
            )
        return np.zeros_like(sensor_range, dtype=np.float64)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        run_test()
        relevant_warnings = [
            w
            for w in recorded_warnings
            if issubclass(w.category, UserWarning)
            and "Performance may be poor when using more than" in str(w.message)
        ]
        assert len(relevant_warnings) == 0


def test_validate_n_sensors_warns_with_ccqr_and_too_many_sensors():
    """Test warning when using CCQR optimizer with more sensors than samples."""
    optimizer = CCQR()
    model = SSPOR(optimizer=optimizer)
    model.basis_matrix_ = np.random.rand(5, 3)
    model.n_sensors = 4

    def test_warning_condition():
        if (
            isinstance(model.optimizer, CCQR)
            and model.n_sensors > model.basis_matrix_.shape[1]
        ):
            warnings.warn(
                "Number of sensors exceeds number of samples, which may cause CCQR to "
                "select sensors in constrained regions."
            )

    with pytest.warns(UserWarning, match="Number of sensors exceeds number of samples"):
        test_warning_condition()


def test_validate_n_sensors_no_warning_when_sensors_within_limit():
    """Test no warning when using CCQR with sensors <= samples."""
    optimizer = CCQR()
    model = SSPOR(optimizer=optimizer)
    model.basis_matrix_ = np.random.rand(5, 3)
    model.n_sensors = 3

    def test_warning_condition():
        if (
            isinstance(model.optimizer, CCQR)
            and model.n_sensors > model.basis_matrix_.shape[1]
        ):
            warnings.warn(
                "Number of sensors exceeds number of samples, which may cause CCQR to "
                "select sensors in constrained regions."
            )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        test_warning_condition()
        relevant_warnings = [
            w
            for w in recorded_warnings
            if issubclass(w.category, UserWarning)
            and "Number of sensors exceeds number of samples" in str(w.message)
        ]
        assert len(relevant_warnings) == 0


def test_set_n_sensors_delegates_to_set_number_of_sensors():
    """Test that set_n_sensors correctly delegates to set_number_of_sensors."""
    model = SSPOR()
    original_method = model.set_number_of_sensors
    model.set_number_of_sensors = Mock()

    try:
        n_sensors_test = 5
        model.set_n_sensors(n_sensors_test)
        model.set_number_of_sensors.assert_called_once_with(n_sensors_test)
        model.set_number_of_sensors.reset_mock()
        n_sensors_test = 10
        model.set_n_sensors(n_sensors_test)
        model.set_number_of_sensors.assert_called_once_with(n_sensors_test)

    finally:
        model.set_number_of_sensors = original_method


def test_set_n_sensors_with_multiple_values():
    """Test set_n_sensors with a variety of values."""
    model = SSPOR()
    calls = []
    original_method = model.set_number_of_sensors
    model.set_number_of_sensors = lambda n: calls.append(n)

    try:
        test_values = [1, 10, 100]
        for value in test_values:
            model.set_n_sensors(value)
        assert calls == test_values

    finally:
        model.set_number_of_sensors = original_method


def test_update_n_basis_modes_raises_error_for_invalid_n_basis_modes():
    """Test update_n_basis_modes raises ValueError for invalid n_basis_modes values."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    invalid_values = [0, -1, 2.5, "10", None]

    for invalid_value in invalid_values:
        with pytest.raises(
            ValueError, match="n_basis_modes must be a positive integer"
        ):
            model.update_n_basis_modes(invalid_value)


def test_update_n_basis_modes_raises_error_when_x_is_none():
    """Test ValueError when n_basis_modes exceeds available and x is None."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    basis.n_basis_modes = 5
    basis.basis_matrix_ = np.random.rand(10, 5)
    with pytest.raises(
        ValueError,
        match="x cannot be None when n_basis_modes exceeds number of available modes",
    ):
        model.update_n_basis_modes(10, x=None)


def test_update_n_basis_modes_raises_error_when_exceeds_examples():
    """Test ValueError when n_basis_modes exceeds number of examples."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    basis.n_basis_modes = 5
    basis.basis_matrix_ = np.random.rand(10, 5)
    x = np.random.rand(8, 10)
    with pytest.raises(
        ValueError, match="n_basis_modes cannot exceed the number of examples"
    ):
        model.update_n_basis_modes(10, x=x)


def test_update_n_basis_modes_no_refit_when_fewer_modes():
    """Test case where n_basis_modes is less than available modes (no error)."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    basis.n_basis_modes = 10
    basis.basis_matrix_ = np.random.rand(20, 10)
    original_fit = model.fit
    model.fit = Mock()

    try:
        model.update_n_basis_modes(5)
        assert model.n_basis_modes == 5
        model.fit.assert_called_once()
        call_args = model.fit.call_args[1]
        assert call_args.get("prefit_basis") is True

    finally:
        model.fit = original_fit


def test_validate_n_sensors_raises_error_when_exceeds_max():
    """Test ValueError when n_sensors exceeds maximum available sensors."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    max_sensors = 5
    model.basis_matrix_ = np.random.rand(max_sensors, 3)
    model.n_sensors = max_sensors + 1
    with pytest.raises(
        ValueError,
        match=f"n_sensors cannot exceed number of available sensors: {max_sensors}",
    ):
        with patch("sklearn.utils.validation.check_is_fitted"):
            model._validate_n_sensors()


def test_validate_n_sensors_sets_default_when_none():
    """Test that n_sensors is set to max_sensors when it's None."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    max_sensors = 5
    model.basis_matrix_ = np.random.rand(max_sensors, 3)
    model.n_sensors = None
    with patch("sklearn.utils.validation.check_is_fitted"):
        model._validate_n_sensors()
    assert model.n_sensors == max_sensors


def test_validate_n_sensors_no_error_when_within_limit():
    """Test that no error is raised when n_sensors is within the limit."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    max_sensors = 5
    model.basis_matrix_ = np.random.rand(max_sensors, 3)
    for n_sensors in [max_sensors, max_sensors - 1, 1]:
        model.n_sensors = n_sensors
        with patch("sklearn.utils.validation.check_is_fitted"):
            model._validate_n_sensors()
        assert model.n_sensors == n_sensors


def test_score_with_custom_score_function():
    """Test that score correctly uses a custom score_function when provided."""
    basis = Mock()
    optimizer = Mock()
    model = SSPOR(basis=basis, optimizer=optimizer)
    model.ranked_sensors_ = np.array([0, 1, 2, 3, 4])
    X = np.random.rand(10, 5)
    selected_sensors = np.array([0, 1, 2])
    model.get_selected_sensors = Mock(return_value=selected_sensors)
    predicted_values = np.random.rand(10, 5)
    model.predict = Mock(return_value=predicted_values)
    calls = []

    def custom_score_function(y_true, y_pred, **kwargs):
        calls.append((y_true, y_pred, kwargs))
        return 0.95

    score_kws = {"beta": 2, "sample_weight": np.ones(10)}
    solve_kws = {"solver": "lstsq"}
    with patch("sklearn.utils.validation.check_is_fitted"):
        result = model.score(
            X,
            score_function=custom_score_function,
            score_kws=score_kws,
            solve_kws=solve_kws,
        )
    assert result == 0.95
    model.predict.assert_called_once()
    predict_args, predict_kwargs = model.predict.call_args
    assert np.array_equal(predict_args[0], X[:, selected_sensors])
    assert "solver" in predict_kwargs
    assert predict_kwargs["solver"] == "lstsq"
    assert len(calls) == 1
    y_true, y_pred, kwargs = calls[0]
    assert np.array_equal(y_true, X)
    assert np.array_equal(y_pred, predicted_values)
    assert "beta" in kwargs
    assert kwargs["beta"] == 2
    assert "sample_weight" in kwargs
    assert np.array_equal(kwargs["sample_weight"], np.ones(10))


def test_reconstruction_error_warning():
    basis_mode_dim = 6
    n_sensors = 8
    model = SSPOR(n_sensors=n_sensors)
    model.basis_matrix_ = np.random.rand(basis_mode_dim, basis_mode_dim)
    model.ranked_sensors_ = np.arange(basis_mode_dim)
    sensor_range = np.arange(1, n_sensors + 1)
    x_test = np.random.rand(5, basis_mode_dim)
    with pytest.warns(
        UserWarning,
        match=f"Performance may be poor when using more than {basis_mode_dim} sensors",
    ):
        model.reconstruction_error(x_test, sensor_range=sensor_range)


def test_validate_n_sensors_warning():
    n_sensors = 10
    n_samples = 5
    model = SSPOR(optimizer=CCQR(), n_sensors=n_sensors)
    model.basis_matrix_ = np.random.rand(15, n_samples)
    with pytest.warns(
        UserWarning,
        match="Number of sensors exceeds number of samples, "
        "which may cause CCQR to select sensors in constrained regions.",
    ):
        model._validate_n_sensors()


def test_std_function():
    X = np.random.rand(5, 10)
    n_basis_modes = 3
    flat_prior = np.full(n_basis_modes, 1)
    model = SSPOR(basis=SVD(n_basis_modes=n_basis_modes))
    model.fit(x=X)
    sigma_flat = model.std(prior=flat_prior, noise=0.1)
    sigma_decreasing = model.std(prior="decreasing", noise=0.1)
    sigmas = [sigma_flat, sigma_decreasing]

    for sigma in sigmas:
        assert sigma is not None
        assert isinstance(sigma, np.ndarray)
        assert sigma.shape == (X.shape[1],)
        assert np.all(sigma >= 0)
        assert not np.any(np.isnan(sigma))


def test_std_none_noise():
    X = np.random.rand(5, 10)
    n_basis_modes = 3
    prior = np.full(n_basis_modes, 1)
    model = SSPOR(basis=SVD(n_basis_modes=n_basis_modes))
    model.fit(x=X)

    # Test with None noise - should trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sigma = model.std(prior=prior, noise=None)  # noqa:F841
        # Check that warning was raised
        assert len(w) == 1
        assert "noise is None" in str(w[0].message)


def test_std_invalid_prior():
    X = np.random.rand(5, 10)
    n_basis_modes = 2
    model = SSPOR(basis=SVD(n_basis_modes=n_basis_modes))
    model.fit(x=X)
    # Invalid string
    with pytest.raises(ValueError):
        model.std(prior="invalid_string", noise=0.1)
    # Invalid 2d prior
    invalid_prior_2d = np.random.rand(2, 2)
    with pytest.raises(ValueError):
        model.std(prior=invalid_prior_2d, noise=0.1)
    # Prior with invalid shape
    wrong_shape_prior = np.random.rand(3)  # Should be length 2
    with pytest.raises(ValueError):
        model.std(prior=wrong_shape_prior, noise=0.1)


def test_std_model_not_fitted():
    model = SSPOR(basis=SVD(n_basis_modes=2))
    prior = np.full(2, 1)

    with pytest.raises(Exception):
        model.std(prior=prior, noise=0.1)


def test_regularized_reconstruction():
    X = np.random.rand(5, 10)
    n_basis_modes = 3
    flat_prior = np.full(n_basis_modes, 1)
    model = SSPOR(basis=SVD(n_basis_modes=n_basis_modes))
    model.fit(x=X)
    selected = model.get_selected_sensors()
    x_sensors = X[:, selected]

    y_pred_flat = model.predict(x_sensors, method=None, prior=flat_prior, noise=0.1)
    y_pred_decreasing = model.predict(
        x_sensors, method=None, prior="decreasing", noise=0.1
    )
    y_preds = [y_pred_flat, y_pred_decreasing]

    for y_pred in y_preds:
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (5, 10)
        assert not np.any(np.isnan(y_pred))
        assert np.isrealobj(y_pred)


def test_regularized_reconstruction_sensor_consistency():
    X = np.random.rand(5, 10)
    n_basis_modes = 3
    prior = np.full(n_basis_modes, 1)
    model = SSPOR(basis=SVD(n_basis_modes=n_basis_modes))
    model.fit(x=X)
    selected = model.get_selected_sensors()
    x_sensors = X[:, selected]

    # Reconstruction should be consistent when using the same sensor data
    y_pred1 = model.predict(x_sensors, method=None, prior=prior, noise=0.1)
    y_pred2 = model.predict(x_sensors, method=None, prior=prior, noise=0.1)

    np.testing.assert_array_equal(y_pred1, y_pred2)


def test_one_pt_landscape():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=3), optimizer=TPGR(n_sensors=3))
    model.fit(x=X)
    flat_prior = np.full(3, 1)

    landscape_flat = model.one_pt_energy_landscape(prior=flat_prior, noise=0.1)
    landscape_decreasing = model.one_pt_energy_landscape(prior="decreasing", noise=0.1)

    landscapes = [landscape_flat, landscape_decreasing]

    for landscape in landscapes:
        assert isinstance(landscape, np.ndarray)
        assert landscape.shape == (10,)
        # One-point landscape should not contain NaN values
        assert not np.any(np.isnan(landscape))
        assert not np.any(np.isinf(landscape))


def test_two_pt_landscape_single_sensor():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=3), optimizer=TPGR(n_sensors=3))
    model.fit(x=X)
    flat_prior = np.full(3, 1)
    selected_sensors = [3]  # Single sensor
    landscape_flat = model.two_pt_energy_landscape(
        selected_sensors=selected_sensors, prior=flat_prior, noise=0.1
    )
    landscape_decreasing = model.two_pt_energy_landscape(
        selected_sensors=selected_sensors, prior="decreasing", noise=0.1
    )
    landscapes = [landscape_flat, landscape_decreasing]

    for landscape in landscapes:
        assert isinstance(landscape, np.ndarray)
        assert landscape.shape == (10,)
        # Selected sensor position should be NaN
        assert np.isnan(landscape[3])
        # Other positions should have finite values
        remaining_mask = np.ones(10, dtype=bool)
        remaining_mask[selected_sensors] = False
        assert not np.any(np.isnan(landscape[remaining_mask]))


def test_two_pt_landscape_multiple_sensors():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=3), optimizer=TPGR(n_sensors=3))
    model.fit(x=X)
    flat_prior = np.full(3, 1)
    selected_sensors = [1, 5, 8]  # Multiple sensors
    landscape_flat = model.two_pt_energy_landscape(
        selected_sensors=selected_sensors, prior=flat_prior, noise=0.1
    )
    landscape_decreasing = model.two_pt_energy_landscape(
        selected_sensors=selected_sensors, prior="decreasing", noise=0.1
    )
    landscapes = [landscape_flat, landscape_decreasing]

    for landscape in landscapes:
        assert isinstance(landscape, np.ndarray)
        assert landscape.shape == (10,)
        # Selected sensor positions should be NaN
        for sensor in selected_sensors:
            assert np.isnan(landscape[sensor])
        # Other positions should have finite values
        remaining_mask = np.ones(10, dtype=bool)
        remaining_mask[selected_sensors] = False
        assert not np.any(np.isnan(landscape[remaining_mask]))


def test_landscapes_none_noise():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=3), optimizer=TPGR(n_sensors=3))
    model.fit(x=X)
    flat_prior = np.full(3, 1)
    selected_sensors = [3]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        landscape = model.one_pt_energy_landscape(  # noqa:F841
            prior=flat_prior, noise=None
        )
        # Check that warning was raised
        assert len(w) == 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        landscape = model.two_pt_energy_landscape(  # noqa:F841
            selected_sensors=selected_sensors, prior=flat_prior, noise=None
        )

        # Check that warning was raised
        assert len(w) == 1
        assert "noise is None" in str(w[0].message)


def test_landscapes_optimizer_requirement():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=3))
    model.fit(x=X)
    flat_prior = np.full(3, 1)
    selected_sensor = [3]

    with pytest.raises(TypeError):
        model.one_pt_energy_landscape(prior=flat_prior, noise=0.1)
    with pytest.raises(TypeError):
        model.two_pt_energy_landscape(
            selected_sensors=selected_sensor, prior=flat_prior, noise=0.1
        )


def test_landscapes_invalid_prior():
    X = np.random.rand(5, 10)
    model = SSPOR(basis=SVD(n_basis_modes=2), optimizer=TPGR(n_sensors=3))
    model.fit(x=X)
    selected_sensors = [1]

    # Invalid string
    with pytest.raises(ValueError):
        model.one_pt_energy_landscape(prior="invalid_string", noise=0.1)
    with pytest.raises(ValueError):
        model.two_pt_energy_landscape(
            selected_sensors=selected_sensors, prior="invalid_string", noise=0.1
        )

    # Invalid 2d prior
    invalid_prior_2d = np.random.rand(2, 2)
    with pytest.raises(ValueError):
        model.one_pt_energy_landscape(prior=invalid_prior_2d, noise=0.1)
    with pytest.raises(ValueError):
        model.two_pt_energy_landscape(
            selected_sensors=selected_sensors, prior=invalid_prior_2d, noise=0.1
        )

    # Prior with invalid shape
    wrong_shape_prior = np.random.rand(3)  # Should be length 2
    with pytest.raises(ValueError):
        model.one_pt_energy_landscape(prior=wrong_shape_prior, noise=0.1)
    with pytest.raises(ValueError):
        model.two_pt_energy_landscape(
            selected_sensors=selected_sensors, prior=wrong_shape_prior, noise=0.1
        )
