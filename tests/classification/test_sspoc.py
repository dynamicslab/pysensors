"""Tests for SSPOC class."""

from unittest.mock import Mock

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import SVD, Identity, RandomProjection
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        (lazy_fixture("data_binary_classification"), 0.55),
        (lazy_fixture("data_multiclass_classification"), 0.25),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        lazy_fixture("data_binary_classification"),
        lazy_fixture("data_multiclass_classification"),
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
        (lazy_fixture("data_binary_classification"), (20,)),
        (lazy_fixture("data_multiclass_classification"), (20, 5)),
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


@pytest.fixture
def sspoc_instance():
    """Create a mock SSPOC instance with refit=False."""
    sspoc = SSPOC()
    sspoc.refit_ = False
    sspoc.n_sensors = 3
    sspoc.basis_matrix_inverse_ = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    sspoc.classifier = Mock()
    sspoc.classifier.predict.return_value = np.array([0, 1, 0])
    sspoc.sensor_coef_ = np.array([1, 2, 3])

    return sspoc


def test_predict_with_refit_false(sspoc_instance):
    """Test predict method when refit is False."""
    X_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected_transformed_input = np.dot(X_test, sspoc_instance.basis_matrix_inverse_.T)
    result = sspoc_instance.predict(X_test)
    sspoc_instance.classifier.predict.assert_called_once()
    actual_arg = sspoc_instance.classifier.predict.call_args[0][0]
    np.testing.assert_array_almost_equal(actual_arg, expected_transformed_input)
    assert np.array_equal(result, np.array([0, 1, 0]))


def test_predict_with_refit_true(sspoc_instance):
    """Test predict method when refit is True."""
    sspoc_instance.refit_ = True
    X_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = sspoc_instance.predict(X_test)
    sspoc_instance.classifier.predict.assert_called_once_with(X_test)
    assert np.array_equal(result, np.array([0, 1, 0]))


def test_predict_with_zero_sensors():
    """Test predict method when n_sensors is 0."""
    sspoc = SSPOC()
    sspoc.n_sensors = 0
    sspoc.sensor_coef_ = np.array([])
    sspoc.dummy_ = Mock()
    sspoc.dummy_.predict.return_value = np.array([1, 1, 1])

    X_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    with pytest.warns(UserWarning, match="SSPOC model has no selected sensors"):
        sspoc.predict(X_test)
    sspoc.dummy_.predict.assert_called_once()
    np.testing.assert_array_equal(sspoc.dummy_.predict.call_args[0][0], X_test[:, 0])


@pytest.fixture
def sspoc_mock():
    """Create a mock SSPOC instance that will actually trigger the warning."""
    sspoc = SSPOC()
    sspoc.sensor_coef_ = np.array([0.9, 0.8, 0.0, 0.3, 0.2])
    sspoc.classifier = Mock()

    def custom_update(
        n_sensors=None,
        threshold=None,
        xy=None,
        quiet=False,
        method=np.max,
        **method_kws,
    ):
        if n_sensors is not None:
            sorted_indices = np.argsort(-np.abs(sspoc.sensor_coef_))
            print(f"Sorted indices: {sorted_indices}")
            print(f"n_sensors-1 index: {sorted_indices[n_sensors - 1]}")
            print(
                f"Value at index: {sspoc.sensor_coef_[sorted_indices[n_sensors - 1]]}"
            )
            print(
                f"Is 0?{np.abs(sspoc.sensor_coef_[sorted_indices[n_sensors - 1]]) == 0}"
            )
        original = sspoc.update_sensors
        return original(n_sensors, threshold, xy, quiet, method, **method_kws)

    return sspoc


@pytest.fixture
def sspoc_multiclass_mock():
    """Create a mock SSPOC instance for multiclass case that will trigger warning."""
    sspoc = SSPOC()
    sspoc.sensor_coef_ = np.array(
        [
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3],
            [0.05, 0.04, 0.03],
        ]
    )
    sspoc.classifier = Mock()
    return sspoc


def test_warning_when_threshold_too_high(sspoc_mock):
    """Test warning when threshold is set too high and no sensors are selected."""
    with pytest.warns(UserWarning, match="Threshold set too high.*no sensors selected"):
        sspoc_mock.update_sensors(threshold=1.0)
    assert len(sspoc_mock.sparse_sensors_) == 0
    assert sspoc_mock.n_sensors == 0


def test_warning_when_no_sensors_selected_for_refit(sspoc_mock):
    """Test warning when trying to refit with no sensors selected."""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    with pytest.warns(UserWarning, match="No selected sensors; model was not refit"):
        sspoc_mock.update_sensors(threshold=1.0, xy=(X, y))
    sspoc_mock.classifier.fit.assert_not_called()


def test_warning_when_both_n_sensors_and_threshold_provided(sspoc_mock):
    """Test that warning is issued when both n_sensors and threshold are provided."""
    with pytest.warns(
        UserWarning,
        match="Both n_sensors.*and threshold.*were passed so threshold will be ignored",
    ):
        sspoc_mock.update_sensors(n_sensors=2, threshold=0.4)


def test_update_sensors_too_many_sensors_error():
    n_available_sensors = 10
    model = SSPOC()
    model.sensor_coef_ = np.random.rand(n_available_sensors)
    too_many_sensors = n_available_sensors + 5

    expected_error = (
        f"n_sensors\\({too_many_sensors}\\) cannot exceed number of "
        f"available sensors \\({n_available_sensors}\\)"
    )

    with pytest.raises(ValueError, match=expected_error):
        model.update_sensors(n_sensors=too_many_sensors)


def test_uninformative_sensors_warning():
    n_available_sensors = 10
    n_sensors_to_select = 6
    model = SSPOC()
    sensor_coef = np.zeros(n_available_sensors)
    sensor_coef[:5] = np.random.rand(5)
    sensor_coef[:5] = -np.sort(-np.abs(sensor_coef[:5]))
    model.sensor_coef_ = sensor_coef
    with pytest.warns(
        UserWarning,
        match="Some uninformative sensors were selected. Consider decreasing n_sensors",
    ):
        model.update_sensors(n_sensors=n_sensors_to_select)


def test_uninformative_sensors_multiclass_warning():
    n_available_sensors = 10
    n_classes = 3
    n_sensors_to_select = 6
    model = SSPOC()
    sensor_coef = np.zeros((n_available_sensors, n_classes))
    sensor_coef[:5, :] = np.random.rand(5, n_classes)
    for i in range(5):
        sensor_coef[i, :] = np.abs(sensor_coef[i, :]) + 0.5
    model.sensor_coef_ = sensor_coef
    with pytest.warns(
        UserWarning,
        match="Some uninformative sensors were selected. Consider decreasing n_sensors",
    ):
        model.update_sensors(n_sensors=n_sensors_to_select, method=np.mean)
