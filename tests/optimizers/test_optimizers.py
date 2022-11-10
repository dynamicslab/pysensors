"""Unit tests for optimizers"""
import numpy as np

from pysensors.optimizers import CCQR
from pysensors.optimizers import QR
from pysensors.optimizers import GQR


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
    sensors_GQR = GQR().fit(x.T, idx_constrained=forbidden_sensors,n_const_sensors=0, constraint_option='exact_n_const_sensors').get_sensors()

    # Forbidden sensors should not be included
    chosen_sensors_GQR = set(sensors_GQR[: (x.shape[1] - len(forbidden_sensors))])
    assert chosen_sensors_GQR.isdisjoint(set(forbidden_sensors))
    assert chosen_sensors_CCQR == chosen_sensors_GQR


def test_gqr_exact_constrainted_case1(data_random):
    ## In this case we want to place a total of 10 sensors
    # with a constrained region that is allowed to have exactly 3 sensors
    # but 4 of the first 10 are in the constrained region
    x = data_random
    # unconstrained sensors (optimal)
    sensors_QR = QR().fit(x.T).get_sensors()
    # exact number of sensors allowed in the constrained region
    total_sensors = 10
    exact_n_const_sensors = 3
    forbidden_sensors = [8,5,2,6]
    totally_forbidden_sensors = [x for x in forbidden_sensors if x in sensors_QR][:exact_n_const_sensors]
    totally_forbidden_sensors = [y for y in forbidden_sensors if y not in totally_forbidden_sensors]
    costs = np.zeros(x.shape[1])
    costs[totally_forbidden_sensors] = 100
    # Get ranked sensors
    sensors = CCQR(sensor_costs=costs).fit(x.T).get_sensors()[:total_sensors]

    # Forbidden sensors should not be included
    chosen_sensors = set(sensors[: (x.shape[1] - len(totally_forbidden_sensors))])
    assert chosen_sensors.isdisjoint(set(totally_forbidden_sensors))


    # Get ranked sensors from GQR
    sensors_GQR = GQR().fit(x.T, idx_constrained=forbidden_sensors,n_sensors=total_sensors,n_const_sensors=exact_n_const_sensors, constraint_option='exact_n_const_sensors').get_sensors()[:total_sensors]

    # try to compare these using the validation metrics

def test_gqr_max_constrained():
    pass

def test_gqr_radii_constrained():
    pass