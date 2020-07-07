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
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysensors import SensorSelector
from pysensors.basis import Identity
from pysensors.basis import POD


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


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_vandermonde"), pytest.lazy_fixture("data_random")],
)
def test_get_all_sensors(data):
    x = data
    max_sensors = x.shape[1]

    model = SensorSelector()
    model.fit(x)
    assert len(model.get_all_sensors()) == max_sensors


@pytest.mark.parametrize("basis", [Identity(), POD()])
def test_basis_compatibility(data_vandermonde, basis):
    x = data_vandermonde
    model = SensorSelector(basis=basis)
    model.fit(x)
    check_is_fitted(model)


# TODO: tests for
#   - predict method
#       Square vs. rectangular matrices (predict method)
#       Wrong size inputs for predict method

# TODO: add more datasets for testing
# TODO: test for accuracy somehow?
