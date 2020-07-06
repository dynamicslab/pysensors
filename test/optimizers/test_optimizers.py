"""Unit tests for optimizers"""
from pysensors.optimizers import QR


def test_num_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    qr = QR()
    sensors = qr.get_sensors(x.T)
    assert len(sensors) == max_sensors
