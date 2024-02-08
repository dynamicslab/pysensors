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
    # first we should pass all_sensors to GQR
    all_sensors = np.arange(x.shape[1]) #QR().fit(x.T).get_sensors()
    sensors_GQR = GQR().fit(x.T, all_sensors=all_sensors, idx_constrained=forbidden_sensors,n_const_sensors=0, constraint_option='exact_n').get_sensors()

    # Forbidden sensors should not be included
    chosen_sensors_GQR = set(sensors_GQR[: (x.shape[1] - len(forbidden_sensors))])
    assert chosen_sensors_GQR.isdisjoint(set(forbidden_sensors))
    assert chosen_sensors_CCQR == chosen_sensors_GQR


def test_gqr_exact_constrainted_case1(data_random):
    ## In this case we want to place a total of 19 sensors
    # with a constrained region that is allowed to have EXACTLY 2 sensors
    # but 3 of the sensors are in the constrained region
    x = data_random
    # unconstrained sensors (optimal)
    sensors_QR = QR().fit(x.T).get_sensors()
    # exact number of sensors allowed in the constrained region
    total_sensors = 19
    exact_n_const_sensors = 2
    forbidden_sensors = list(sensors_QR[[7,11,-1]])
    totally_forbidden_sensors = [x for x in forbidden_sensors if x in sensors_QR][:exact_n_const_sensors]
    totally_forbidden_sensors = [y for y in forbidden_sensors if y not in totally_forbidden_sensors]
    costs = np.zeros(x.shape[1])
    costs[totally_forbidden_sensors] = 100
    # Get ranked sensors
    sensors_CCQR = CCQR(sensor_costs=costs).fit(x.T).get_sensors()[:total_sensors]

    # Forbidden sensors should not be included
    assert set(sensors_CCQR).isdisjoint(set(totally_forbidden_sensors))
    


    # Get ranked sensors from GQR
    sensors_GQR = GQR().fit(x.T, idx_constrained=forbidden_sensors,all_sensors=sensors_QR, n_sensors=total_sensors,n_const_sensors=exact_n_const_sensors, constraint_option='exact_n').get_sensors()[:total_sensors]
    assert sensors_CCQR.all() == sensors_GQR.all()
    

## TODO
def test_gqr_max_constrained():
    pass

def test_gqr_radii_constrained():
    pass