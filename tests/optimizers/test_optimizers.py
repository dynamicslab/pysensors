"""Unit tests for optimizers"""
import numpy as np

from pysensors.optimizers import CCQR
from pysensors.optimizers import QR


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
