"""Tests for SSPOC class."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import Identity
from pysensors.basis import RandomProjection
from pysensors.basis import SVD
from pysensors.classification import SSPOC


SEED = 15


@pytest.fixture
def data_binary_classification():
    x, y = make_classification(n_classes=2, random_state=SEED)
    l1_penalty = 1
    return x, y, l1_penalty


@pytest.fixture
def data_multiclass_classification():
    x, y = make_classification(n_classes=5, n_informative=5, random_state=SEED)
    l1_penalty = 0.03
    return x, y, l1_penalty


def test_not_fitted(data_binary_classification):
    x, y, _ = data_binary_classification
    model = SSPOC()

    # Shouldn't be able to call any of these methods before fitting
    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.update_sensors(n_sensors=5)
    with pytest.raises(NotFittedError):
        model.selected_sensors


def test_prefit_basis(data_binary_classification):
    x, y, _ = data_binary_classification
    basis = Identity().fit(x)
    model_prefit = SSPOC(basis=basis)
    model_prefit.fit(x, y, prefit_basis=True, quiet=True)

    model_standard = SSPOC().fit(x, y, quiet=True)

    np.testing.assert_allclose(model_prefit.sensor_coef_, model_standard.sensor_coef_)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_initialize_with_n_sensors(data):
    x, y, l1_penalty = data
    n_sensors = 3
    model = SSPOC(n_sensors=n_sensors, l1_penalty=l1_penalty).fit(x, y, quiet=True)

    assert len(model.selected_sensors) == n_sensors
    assert model.n_sensors == n_sensors


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_initialize_with_threshold(data):
    x, y, l1_penalty = data
    max_sensors = x.shape[1]
    model = SSPOC(threshold=0, l1_penalty=l1_penalty).fit(x, y, quiet=True)

    assert len(model.selected_sensors) == max_sensors
    assert model.n_sensors == max_sensors


@pytest.mark.parametrize("n_sensors", [3, 0])
@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_update_n_sensors(data, n_sensors):
    x, y, l1_penalty = data
    model = SSPOC(l1_penalty=l1_penalty).fit(x, y, quiet=True)

    model.update_sensors(n_sensors=n_sensors, quiet=True)
    assert len(model.selected_sensors) == n_sensors
    assert model.n_sensors == n_sensors


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_update_threshold(data):
    x, y, l1_penalty = data

    model = SSPOC(threshold=0.01, l1_penalty=l1_penalty).fit(x, y, quiet=True)
    nnz = len(model.selected_sensors)

    # Larger threshold should result in fewer sensors
    model.update_sensors(threshold=1, quiet=True)
    assert len(model.selected_sensors) < nnz


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_large_threshold(data):
    x, y, l1_penalty = data
    model = SSPOC(l1_penalty=l1_penalty).fit(x, y, quiet=True)

    model.update_sensors(threshold=10, quiet=True)
    assert len(model.selected_sensors) == 0
    assert model.n_sensors == 0


def test_bad_update_sensors_input(data_binary_classification):
    x, y, _ = data_binary_classification
    model = SSPOC().fit(x, y, quiet=True)

    with pytest.raises(ValueError):
        model.update_sensors()


@pytest.mark.parametrize(
    "data, baseline_accuracy",
    [
        (pytest.lazy_fixture("data_binary_classification"), 0.55),
        (pytest.lazy_fixture("data_multiclass_classification"), 0.25),
    ],
)
def test_predict_accuracy(data, baseline_accuracy):
    x, y, l1_penalty = data
    model = SSPOC(threshold=0, l1_penalty=l1_penalty).fit(x, y, quiet=True)

    assert (
        accuracy_score(y, model.predict(x[:, model.selected_sensors]))
        > baseline_accuracy
    )


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
def test_dummy_predict(data):
    x, y, l1_penalty = data
    model = SSPOC(l1_penalty=l1_penalty).fit(x, y, quiet=True)
    model.update_sensors(n_sensors=0, xy=(x, y), quiet=True)

    assert model.n_sensors == 0
    # Test that model can still make predictions, albeit random ones
    # when it has no sensors to work with
    y_pred = model.predict(x)
    assert len(y_pred) == len(y)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary_classification"),
        pytest.lazy_fixture("data_multiclass_classification"),
    ],
)
@pytest.mark.parametrize(
    "basis", [Identity(), SVD(), RandomProjection(n_basis_modes=5)]
)
def test_basis_integration(basis, data):
    x, y, _ = data
    model = SSPOC(basis=basis, n_sensors=5)
    model.fit(x, y, quiet=True)

    check_is_fitted(model)


@pytest.mark.parametrize(
    "data, shape",
    [
        (pytest.lazy_fixture("data_binary_classification"), (20,)),
        (pytest.lazy_fixture("data_multiclass_classification"), (20, 5)),
    ],
)
def test_coefficient_shape(data, shape):
    x, y, _ = data
    model = SSPOC().fit(x, y, quiet=True)

    assert model.sensor_coef_.shape == shape


@pytest.mark.parametrize("basis", [SVD, RandomProjection])
def test_update_n_basis_modes_errors(basis, data_binary_classification):
    x, y, _ = data_binary_classification
    n_basis_modes = 5
    model = SSPOC(basis=basis(n_basis_modes=n_basis_modes))

    model.fit(x, y, quiet=True)

    with pytest.raises(ValueError):
        model.update_n_basis_modes(0, xy=(x, y))
    with pytest.raises(ValueError):
        model.update_n_basis_modes("5", xy=(x, y))
    with pytest.raises(ValueError):
        model.update_n_basis_modes(x.shape[0] + 1, xy=(x, y))


@pytest.mark.parametrize("basis", [SVD, RandomProjection])
def test_update_n_basis_modes_shape(basis, data_binary_classification):
    x, y, _ = data_binary_classification
    n_basis_modes_init = 10
    model = SSPOC(basis=basis(n_basis_modes=n_basis_modes_init))
    model.fit(x, y, quiet=True)
    assert model.basis.n_basis_modes == n_basis_modes_init
    assert model.basis_matrix_inverse_.shape[0] == n_basis_modes_init

    n_basis_modes = 5
    model.update_n_basis_modes(n_basis_modes, xy=(x, y), quiet=True)
    assert model.basis.n_basis_modes == n_basis_modes_init
    assert model.basis_matrix_inverse_.shape[0] == n_basis_modes


@pytest.mark.parametrize("basis", [SVD, RandomProjection])
def test_update_n_basis_modes_refit(basis, data_binary_classification):
    x, y, _ = data_binary_classification
    n_basis_modes = 5
    model = SSPOC(basis=basis(n_basis_modes=n_basis_modes))
    model.fit(x, y, quiet=True)
    assert model.basis_matrix_inverse_.shape[0] == n_basis_modes

    model.update_n_basis_modes(n_basis_modes + 1, (x, y), quiet=True)
    assert model.basis_matrix_inverse_.shape[0] == n_basis_modes + 1


@pytest.mark.parametrize("basis", [SVD, RandomProjection])
def test_update_n_basis_modes_unfit_basis(basis, data_binary_classification):
    x, y, _ = data_binary_classification
    n_basis_modes = 5
    model = SSPOC(basis=basis())
    model.update_n_basis_modes(n_basis_modes, (x, y), quiet=True)

    assert model.basis_matrix_inverse_.shape[0] == n_basis_modes


def test_sspoc_selector_equivalence(data_multiclass_classification):
    x, y, _ = data_multiclass_classification

    model = SSPOC().fit(x, y)

    np.testing.assert_array_equal(model.get_selected_sensors(), model.selected_sensors)
