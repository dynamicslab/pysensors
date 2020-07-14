"""
SensorSelector object definition.
"""
from warnings import warn

import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import Identity
from pysensors.optimizers import QR
from pysensors.utils import validate_input


INT_TYPES = (int, np.int64, np.int32, np.int16, np.int8)


class SensorSelector(BaseEstimator):
    """TODO: write docstring

    <Description>

    Parameters
    ----------
    basis: basis object, optional
        Basis in which to represent the data. Default is the identity basis
        (i.e. raw features).

    optimizer: optimizer object, optional
        Optimization method used to identify sparse sensors.

    n_sensors: int, optional (default n_input_features)
        Number of sensors to select. Note that
        ``s = SensorSelector(n_sensors=10); s.fit(x)``
        is equivalent to
        ``s = SensorSelector(); s.fit(x); s.set_number_of_sensors(10)``.

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(self, basis=None, optimizer=None, n_sensors=None):
        if basis is None:
            basis = Identity()
        self.basis = basis
        if optimizer is None:
            optimizer = QR()
        self.optimizer = optimizer
        if n_sensors is None:
            self.n_sensors = None
        elif isinstance(n_sensors, INT_TYPES) and n_sensors > 0:
            self.n_sensors = n_sensors
        else:
            raise ValueError("n_sensors must be a positive integer.")

    def fit(self, x, **optimizer_kws):
        """
        Fit the SensorSelector model, determining which sensors are relevant.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Training data.

        optimizer_kws: dict
            Keyword arguments to be passed to the `get_sensors` method of the optimizer.
        """

        # TODO: some kind of preprocessing / quality control on x
        x = validate_input(x)

        # Fit basis functions to data (sometimes unnecessary, e.g FFT)
        self.basis.fit(x)

        # Get matrix representation of basis
        self.basis_matrix_ = self.basis.matrix_representation()

        # Maximum number of sensors (= dimension of basis vectors)
        max_sensors = self.basis_matrix_.shape[0]
        if self.n_sensors is None:
            self.n_sensors = max_sensors
        elif self.n_sensors > max_sensors:
            raise ValueError(
                "n_sensors cannot exceed number of available sensors: {}".format(
                    max_sensors
                )
            )

        # Find sparse sensor locations
        self.selected_sensors_ = self.optimizer.get_sensors(
            self.basis_matrix_, **optimizer_kws
        )

    def predict(self, x, **solve_kws):
        """
        TODO: docstring

        If x is a column vector, should behave fine.
        If x is a 2D array with rows corresponding to examples we'll need
        to transpose it before multiplying it with the basis matrix.
        """
        check_is_fitted(self, "selected_sensors_")
        x = validate_input(x, self.selected_sensors_[: self.n_sensors]).T

        # For efficiency we may want to factor
        # self.basis_matrix_[self.selected_sensors_, :]
        # in case predict is called multiple times

        if self.n_sensors > self.basis_matrix_.shape[0]:
            warn("n_sensors exceeds dimension of basis modes. Performance may be poor")

        # Square matrix
        if self.n_sensors == self.basis_matrix_.shape[1]:
            return self._square_predict(
                x, self.selected_sensors_[: self.n_sensors], **solve_kws
            )
        # Rectangular matrix
        else:
            return self._rectangular_predict(
                x, self.selected_sensors_[: self.n_sensors], **solve_kws
            )

    def _square_predict(self, x, sensors, **solve_kws):
        return np.dot(
            self.basis_matrix_, solve(self.basis_matrix_[sensors, :], x, **solve_kws)
        ).T

    def _rectangular_predict(self, x, sensors, **solve_kws):
        return np.dot(
            self.basis_matrix_, lstsq(self.basis_matrix_[sensors, :], x, **solve_kws)[0]
        ).T

    def get_selected_sensors(self):
        check_is_fitted(self, "selected_sensors_")
        return self.selected_sensors_[: self.n_sensors]

    def get_all_sensors(self):
        check_is_fitted(self, "selected_sensors_")
        return self.selected_sensors_

    # TODO: functionality for selecting how many sensors to use
    def set_number_of_sensors(self, n_sensors):
        check_is_fitted(self, "selected_sensors_")

        if not isinstance(n_sensors, INT_TYPES):
            raise ValueError("n_sensors must be a positive integer")
        elif n_sensors <= 0:
            raise ValueError("n_sensors must be a positive integer")
        elif n_sensors > len(self.selected_sensors_):
            raise ValueError(
                "n_sensors cannot exceed number of available sensors: "
                "{}".format(len(self.selected_sensors_))
            )
        else:
            self.n_sensors = n_sensors

    def reconstruction_error(self, x_test, sensor_range=None, score=None, **solve_kws):
        """
        Compute the reconstruction error for different numbers of sensors.

        TODO: write docstring
        """
        check_is_fitted(self, "selected_sensors_")
        x_test = validate_input(x_test, self.selected_sensors_[: self.n_sensors]).T

        basis_mode_dim, n_basis_modes = self.basis_matrix_.shape
        if sensor_range is None:
            sensor_range = np.arange(1, min(self.n_sensors, basis_mode_dim) + 1)
        if sensor_range[-1] > basis_mode_dim:
            warn(
                f"Performance may be poor when using more than {basis_mode_dim} sensors"
            )

        if score is None:

            def score(x, y):
                return np.sqrt(np.mean((x - y) ** 2))

        error = np.zeros_like(sensor_range, dtype=np.float64)

        for k, n_sensors in enumerate(sensor_range):
            if n_sensors == n_basis_modes:
                error[k] = score(
                    self._square_predict(
                        x_test[self.selected_sensors_[:n_sensors]],
                        self.selected_sensors_[:n_sensors],
                        **solve_kws,
                    ),
                    x_test.T,
                )
            else:
                error[k] = score(
                    self._rectangular_predict(
                        x_test[self.selected_sensors_[:n_sensors]],
                        self.selected_sensors_[:n_sensors],
                        **solve_kws,
                    ),
                    x_test.T,
                )

        return error
