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
import pytest
from numpy import isnan
from numpy import mean
from numpy import nan
from numpy import sqrt
from numpy import zeros
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import Identity
from pysensors.basis import RandomProjection
from pysensors.basis import SVD
from pysensors.optimizers import CCQR
from pysensors.reconstruction import SSPOR


def test_not_fitted(data_vandermonde):
    x = data_vandermonde
    model = SSPOR()

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
    [pytest.lazy_fixture("data_vandermonde"), pytest.lazy_fixture("data_random")],
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
    assert sqrt(mean((x_test - model.predict(x_test[sensors])) ** 2)) <= 1.0e-3

    # Should also work for row vectors
    x_test = x_test.reshape(1, -1)
    assert sqrt(mean((x_test - model.predict(x_test[:, sensors])) ** 2)) <= 1.0e-3


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
    model.update_n_basis_modes(n_basis_modes)
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
