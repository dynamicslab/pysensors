"""
SensorSelector object definition.
"""
import warnings

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
    """
    A model for selecting the best sensor locations for reconstruction or
    classification tasks.

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
            self.n_sensors = int(n_sensors)
        else:
            raise ValueError("n_sensors must be a positive integer.")

    def fit(self, x, quiet=False, **optimizer_kws):
        """
        Fit the SensorSelector model, determining which sensors are relevant.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Training data.

        quiet: boolean, optional (default False)
            Whether or not to suppress warnings during fitting.

        optimizer_kws: dict, optional
            Keyword arguments to be passed to the `get_sensors` method of the optimizer.
        """

        x = validate_input(x)

        # Fit basis functions to data (sometimes unnecessary, e.g FFT)
        action = "ignore" if quiet else "default"
        with warnings.catch_warnings():
            warnings.filterwarnings(action, category=UserWarning)
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
        self.ranked_sensors_ = self.optimizer.get_sensors(
            self.basis_matrix_, **optimizer_kws
        )

    def predict(self, x, **solve_kws):
        """
        Predict values at all positions given measurements at sensor locations.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_sensors)
            Measurements from which to form prediction.
            The measurements should be taken at the sensor locations specified by
            `self.get_ranked_sensors()`.

        solve_kws: dict, optional
            keyword arguments to be passed to the linear solver used to invert
            the basis matrix.

        Returns
        -------
        y: numpy array, shape (n_samples, n_features)
            Predicted values at every location.
        """
        check_is_fitted(self, "ranked_sensors_")
        x = validate_input(x, self.ranked_sensors_[: self.n_sensors]).T

        # For efficiency we may want to factor
        # self.basis_matrix_[self.ranked_sensors_, :]
        # in case predict is called multiple times.
        # Although if the user changes the number of sensors between calls
        # the factorization will be wasted.

        if self.n_sensors > self.basis_matrix_.shape[0]:
            warnings.warn(
                "n_sensors exceeds dimension of basis modes. Performance may be poor"
            )

        # Square matrix
        if self.n_sensors == self.basis_matrix_.shape[1]:
            return self._square_predict(
                x, self.ranked_sensors_[: self.n_sensors], **solve_kws
            )
        # Rectangular matrix
        else:
            return self._rectangular_predict(
                x, self.ranked_sensors_[: self.n_sensors], **solve_kws
            )

    def _square_predict(self, x, sensors, **solve_kws):
        """Get prediction when the problem is square."""
        return np.dot(
            self.basis_matrix_, solve(self.basis_matrix_[sensors, :], x, **solve_kws)
        ).T

    def _rectangular_predict(self, x, sensors, **solve_kws):
        """Get prediction when the problem is rectangular."""
        return np.dot(
            self.basis_matrix_, lstsq(self.basis_matrix_[sensors, :], x, **solve_kws)[0]
        ).T

    def get_selected_sensors(self):
        """
        Get the indices of the sensors chosen by the model.

        Returns
        -------
        sensors: numpy array, shape (n_sensors,)
            Indices of the sensors chosen by the model
            (i.e. the sensor locations) ranked in descending order
            of importance.
        """
        check_is_fitted(self, "ranked_sensors_")
        return self.ranked_sensors_[: self.n_sensors]

    def get_all_sensors(self):
        """
        Get a ranked list consisting of all the sensors.
        The sensors are given in descending order of importance.

        Returns
        -------
        sensors: numpy array, shape (n_features,)
            Indices of sensors in descending order of importance.
        """
        check_is_fitted(self, "ranked_sensors_")
        return self.ranked_sensors_

    def set_number_of_sensors(self, n_sensors):
        """
        Set `n_sensors`, the number of sensors to be used for prediction.

        Parameters
        ----------
        n_sensors: int
            The number of sensors. Must be a positive integer.
            Cannot exceed the number of available sensors (n_features).
        """
        check_is_fitted(self, "ranked_sensors_")

        if not isinstance(n_sensors, INT_TYPES):
            raise ValueError("n_sensors must be a positive integer")
        elif n_sensors <= 0:
            raise ValueError("n_sensors must be a positive integer")
        elif n_sensors > len(self.ranked_sensors_):
            raise ValueError(
                "n_sensors cannot exceed number of available sensors: "
                "{}".format(len(self.ranked_sensors_))
            )
        else:
            self.n_sensors = n_sensors

    def score(self, x, y=None, score_function=None, score_kws={}, solve_kws={}):
        """
        Compute the reconstruction error for a given set of measurements.

        Parameters
        ----------
        x: numpy array, shape (n_examples, n_features)
            Measurements with which to compute the score.
            Note that `x` should consist of measurements at every location,
            not just the recommended sensor location, i.e. its shape should be
            (n_examples, n_features) rather than (n_examples, n_sensors).

        y: None
            Dummy input to maintain compatibility with Scikit-learn.

        score_function: callable, optional (default None)
            Function used to compute the score. Should have the call signature
            `score_function(y_true, y_pred, **score_kws)`.
            Default is the negative of the root mean squared error
            (sklearn expects higher scores to correspond to better performance).

        score_kws: dict, optional
            Keyword arguments to be passed to score_function. Ignored if
            score_function is None.

        solve_kws: dict, optional
            Keyword arguments to be passed to the predict method.

        Returns
        -------
        score: float
            The score.
        """
        check_is_fitted(self, "ranked_sensors_")

        n_input_features = len(x) if np.ndim(x) == 1 else x.shape[1]
        n_expected_features = len(self.ranked_sensors_)
        if n_expected_features != n_input_features:
            raise ValueError(
                f"x has {n_input_features} features (columns), "
                f"but should have {n_expected_features}"
            )

        sensors = self.get_selected_sensors()
        if score_function is None:
            return -np.sqrt(
                np.mean((self.predict(x[:, sensors], **solve_kws) - x) ** 2)
            )
        else:
            return score_function(
                x, self.predict(x[:, sensors], **solve_kws), **score_kws,
            )

    def reconstruction_error(self, x_test, sensor_range=None, score=None, **solve_kws):
        """
        Compute the reconstruction error for different numbers of sensors.

        Parameters
        ----------
        x_test: numpy array, shape (n_examples, n_features)
            Measurements to be reconstructed.

        sensor_range: 1D numpy array, optional (default None)
            Numbers of sensors at which to compute the reconstruction error.
            If None, will be set to
            [1, 2, ... , min(`n_sensors`, `basis.n_basis_modes`)].

        score: callable, optional (default None)
            Function used to compute the reconstruction error.
            Should have the signature `score(x, x_pred)`.
            If None, the root mean squared error is used.

        solve_kws: dict, optional
            Keyword arguments to be passed to the linear solver.

        Returns
        -------
        error: numpy array, shape (len(sensor_range),)
            Reconstruction scores for each number of sensors in `sensor_range`.
        """
        check_is_fitted(self, "ranked_sensors_")
        x_test = validate_input(x_test, self.get_selected_sensors()).T

        basis_mode_dim, n_basis_modes = self.basis_matrix_.shape
        if sensor_range is None:
            sensor_range = np.arange(1, min(self.n_sensors, basis_mode_dim) + 1)
        if sensor_range[-1] > basis_mode_dim:
            warnings.warn(
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
                        x_test[self.ranked_sensors_[:n_sensors]],
                        self.ranked_sensors_[:n_sensors],
                        **solve_kws,
                    ),
                    x_test.T,
                )
            else:
                error[k] = score(
                    self._rectangular_predict(
                        x_test[self.ranked_sensors_[:n_sensors]],
                        self.ranked_sensors_[:n_sensors],
                        **solve_kws,
                    ),
                    x_test.T,
                )

        return error
