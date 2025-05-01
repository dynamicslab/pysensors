"""Unit tests for optimizers"""

import numpy as np
import pytest

from pysensors.optimizers import CCQR, GQR, QR, qr_reflector


def test_num_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    qr = QR()
    sensors = qr.fit(x.T).get_sensors()
    assert len(sensors) == max_sensors


def test_ccqr_qr_equivalence(data_vandermonde):
    x = data_vandermonde

    qr_sensors = QR().fit(x.T).get_sensors()
    # If no costs are passed, all zeros are used
    ccqr_sensors = CCQR().fit(x.T).get_sensors()

    np.testing.assert_array_equal(qr_sensors, ccqr_sensors)


def test_ccqr_sensor_placement(data_random):
    x = data_random

    forbidden_sensors = np.arange(0, x.shape[1], 3)
    costs = np.zeros(x.shape[1])
    costs[forbidden_sensors] = 100
    # Get ranked sensors
    sensors = CCQR(sensor_costs=costs).fit(x.T).get_sensors()

    # Forbidden sensors should not be included
    chosen_sensors = set(sensors[: (x.shape[1] - len(forbidden_sensors))])
    assert chosen_sensors.isdisjoint(set(forbidden_sensors))


def test_ccqr_negative_costs(data_vandermonde):
    x = data_vandermonde

    desirable_sensors = np.array([20, 55, 99, 100, 150])
    costs = np.zeros(x.shape[1])
    costs[desirable_sensors] = -100

    sensors = CCQR(sensor_costs=costs).fit(x.T).get_sensors()

    chosen_sensors = set(sensors[: min(x.shape)])
    assert all(s in chosen_sensors for s in set(desirable_sensors))


def test_gqr_qr_equivalence(data_vandermonde):
    x = data_vandermonde

    gqr_sensors = GQR().fit(x.T).get_sensors()
    # If no constraints are passed it should converge to the regular QR optimizer
    qr_sensors = QR().fit(x.T).get_sensors()

    np.testing.assert_array_equal(gqr_sensors, qr_sensors)


def test_gqr_ccqr_equivalence(data_random):
    x = data_random

    forbidden_sensors = np.arange(0, x.shape[1], 3)
    costs = np.zeros(x.shape[1])
    costs[forbidden_sensors] = 100
    # Get ranked sensors from CCQR
    sensors = CCQR(sensor_costs=costs).fit(x.T).get_sensors()

    # Forbidden sensors should not be included
    chosen_sensors_CCQR = set(sensors[: (x.shape[1] - len(forbidden_sensors))])
    assert chosen_sensors_CCQR.isdisjoint(set(forbidden_sensors))

    # Get ranked sensors from GQR
    # first we should pass all_sensors to GQR
    all_sensors = np.arange(x.shape[1])  # QR().fit(x.T).get_sensors()
    sensors_GQR = (
        GQR()
        .fit(
            x.T,
            all_sensors=all_sensors,
            idx_constrained=forbidden_sensors,
            n_const_sensors=0,
            constraint_option="exact_n",
        )
        .get_sensors()
    )

    # Forbidden sensors should not be included
    chosen_sensors_GQR = set(sensors_GQR[: (x.shape[1] - len(forbidden_sensors))])
    assert chosen_sensors_GQR.isdisjoint(set(forbidden_sensors))
    assert chosen_sensors_CCQR == chosen_sensors_GQR


def test_gqr_exact_constrainted_case1(data_random):
    x = data_random
    # unconstrained sensors (optimal)
    sensors_QR = QR().fit(x.T).get_sensors()
    # exact number of sensors allowed in the constrained region
    total_sensors = 19
    exact_n_const_sensors = 2
    forbidden_sensors = list(sensors_QR[[7, 11, -1]])
    totally_forbidden_sensors = [x for x in forbidden_sensors if x in sensors_QR][
        :exact_n_const_sensors
    ]
    totally_forbidden_sensors = [
        y for y in forbidden_sensors if y not in totally_forbidden_sensors
    ]
    costs = np.zeros(x.shape[1])
    costs[totally_forbidden_sensors] = 100
    # Get ranked sensors
    sensors_CCQR = CCQR(sensor_costs=costs).fit(x.T).get_sensors()[:total_sensors]

    # Forbidden sensors should not be included
    assert set(sensors_CCQR).isdisjoint(set(totally_forbidden_sensors))

    # Get ranked sensors from GQR
    sensors_GQR = (
        GQR()
        .fit(
            x.T,
            idx_constrained=forbidden_sensors,
            all_sensors=sensors_QR,
            n_sensors=total_sensors,
            n_const_sensors=exact_n_const_sensors,
            constraint_option="exact_n",
        )
        .get_sensors()[:total_sensors]
    )
    assert sensors_CCQR.all() == sensors_GQR.all()


def test_qr_reflector_zero_norm():
    """Test qr_reflector when column norms are zero (forcing the else branch)."""
    n_features = 3
    n_examples = 4
    r = np.zeros((n_features, n_examples))
    costs = np.array([0.1, 0.2, 0.3, 0.4])
    u, i_piv = qr_reflector(r, costs)
    assert i_piv == 0
    assert u.shape == (n_features,)
    assert u[0] == np.sqrt(2)
    assert np.all(u[1:] == 0)


def test_qr_reflector_with_positive_norm():
    """Test qr_reflector normal behavior with positive column norms."""
    r = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]).T
    costs = np.array([0.5, 0.5, 0.5])
    u, i_piv = qr_reflector(r, costs)
    assert i_piv == 2
    expected_u0 = 0
    if expected_u0 == 0:
        expected_u0 = 1
    expected_u = np.array([expected_u0, 0, 1]) / np.sqrt(np.abs(expected_u0))
    np.testing.assert_almost_equal(u, expected_u)


def test_qr_reflector_cost_affects_pivot():
    """Test that the cost function properly influences pivot selection."""
    r = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]).T
    costs = np.array([0.0, 0.0, 2.0])
    u, i_piv = qr_reflector(r, costs)
    assert i_piv == 1
    expected_u = np.array([1, 1, 0]) / np.sqrt(1)
    np.testing.assert_almost_equal(u, expected_u)


def test_ccqr_init_raises_error_for_non_1d_sensor_costs():
    """Test that CCQR raises ValueError when sensor_costs is not 1D."""
    sensor_costs_2d = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError) as excinfo:
        CCQR(sensor_costs=sensor_costs_2d)
    error_msg = str(excinfo.value)
    assert "sensor_costs must be a 1D array" in error_msg
    assert "2D array was given" in error_msg
    sensor_costs_0d = np.array(5)
    with pytest.raises(ValueError) as excinfo:
        CCQR(sensor_costs=sensor_costs_0d)
    error_msg = str(excinfo.value)
    assert "sensor_costs must be a 1D array" in error_msg
    assert "0D array was given" in error_msg
    sensor_costs_3d = np.zeros((2, 3, 4))
    with pytest.raises(ValueError) as excinfo:
        CCQR(sensor_costs=sensor_costs_3d)
    error_msg = str(excinfo.value)
    assert "sensor_costs must be a 1D array" in error_msg
    assert "3D array was given" in error_msg
    sensor_costs_1d = np.array([1, 2, 3, 4])
    ccqr = CCQR(sensor_costs=sensor_costs_1d)
    assert ccqr.sensor_costs is sensor_costs_1d
    ccqr = CCQR(sensor_costs=None)
    assert ccqr.sensor_costs is None


def test_ccqr_fit_raises_error_for_mismatched_dimensions():
    """Test CCQR.fit raise ValueError when sensor_costs dimension doesn't match data."""
    sensor_costs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    ccqr = CCQR(sensor_costs=sensor_costs)
    basis_matrix = np.random.rand(3, 4)
    with pytest.raises(ValueError) as excinfo:
        ccqr.fit(basis_matrix)
    error_msg = str(excinfo.value)
    assert "Dimension of sensor_costs (5)" in error_msg
    assert "does not match number of sensors in data (3)" in error_msg
    sensor_costs = np.array([0.1, 0.2])
    ccqr = CCQR(sensor_costs=sensor_costs)
    basis_matrix = np.random.rand(4, 3)
    with pytest.raises(ValueError) as excinfo:
        ccqr.fit(basis_matrix)
    error_msg = str(excinfo.value)
    assert "Dimension of sensor_costs (2)" in error_msg
    assert "does not match number of sensors in data (4)" in error_msg
    sensor_costs = np.array([0.1, 0.2, 0.3, 0.4])
    ccqr = CCQR(sensor_costs=sensor_costs)
    basis_matrix = np.random.rand(4, 5)
    ccqr.fit(basis_matrix)
    assert hasattr(ccqr, "pivots_")
    assert ccqr.pivots_.shape == (4,)
