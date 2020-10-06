"""Unit tests for optimizers"""
import numpy as np

from pysensors.optimizers import CCQR
from pysensors.optimizers import QR


def test_num_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    qr = QR()
    sensors = qr.get_sensors(x.T)
    assert len(sensors) == max_sensors


def test_ccqr_qr_equivalence(data_vandermonde):
    x = data_vandermonde

    qr_sensors = QR().get_sensors(x.T)

    costs = np.zeros(x.shape[1])
    ccqr_senors = CCQR(sensor_costs=costs).get_sensors(x.T)

    np.testing.assert_array_equal(qr_sensors, ccqr_senors)
