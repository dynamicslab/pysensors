"""
Unit tests for SensorSelector class (and any other classes implemented
in pysensors.py).

Note: all tests should be encapsulated in functions whose
names start with "test_"

To run all tests for this package, navigate to the top-level
directory and execute the following command:
pytest

To run tests for just one file, run
pytest file_to_test.py
"""
from sklearn.exceptions import NotFittedError
import pytest

from pysensors import SensorSelector


def test_not_fitted(data_vandermonde):
    x = data_vandermonde
    model = SensorSelector()

    # Should not be able to call any of these methods before fitting
    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.get_selected_sensors()
    with pytest.raises(NotFittedError):
        model.get_all_sensors()
    with pytest.raises(NotFittedError):
        model.set_number_of_sensors(20)


def test_set_number_of_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    model = SensorSelector()
    model.fit(x)

    with pytest.raises(ValueError):
        model.set_number_of_sensors(max_sensors + 1)
    with pytest.raises(ValueError):
        model.set_number_of_sensors(-1)

    model.set_number_of_sensors(15)
    assert len(model.get_selected_sensors()) == 15


def test_get_all_sensors(data_vandermonde):
    x = data_vandermonde
    max_sensors = x.shape[1]

    model = SensorSelector()
    model.fit(x)
    assert len(model.get_all_sensors()) == max_sensors


# TODO: tests for
#   - predict method
#       Square vs. rectangular matrices (predict method)
#       Wrong size inputs for predict method

# TODO: add more datasets for testing
# TODO: test for accuracy somehow?