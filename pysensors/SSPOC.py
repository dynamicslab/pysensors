"""
SSPOC object definition.
"""
import warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .basis import Identity
from .optimizers import LDA
from .pysensors import SensorSelector
from .utils import validate_input


INT_TYPES = (int, np.int64, np.int32, np.int16, np.int8)


class SSPOC(SensorSelector):
    """
    Sparse Sensor Placement Optimization for Classification.
    """

    def __init__(self, basis=None, optimizer=None, n_sensors=None):
        if optimizer is None:
            optimizer = CVX
        super(SSPOC, self).__init__(
            basis=basis, optimizer=optimizer, n_sensors=n_sensors
        )

    def fit(self, x, y, quiet=False, prefit_basis=False, seed=None, **optimizer_kws):
        """
        Fit the SSPOC model, determining which sensors are relevant.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Training data.

        y: array-like, shape (n_samples,)
            Training labels.

        quiet: boolean, optional (default False)
            Whether or not to suppress warnings during fitting.

        prefit_basis: boolean, optional (default False)
            Whether or not the basis has already been fit to x.
            For example, you may have already fit and experimented with
            a ``POD`` object to determine the optimal number of modes. This
            option allows you to avoid an unnecessary SVD.

        seed: int, optional (default None)
            Seed for the random number generator used to shuffle sensors after the
            ``self.basis.n_basis_modes`` sensor. Most optimizers only rank the top
            ``self.basis.n_basis_modes`` sensors, leaving the rest virtually
            untouched. As a result the remaining samples are randomly permuted.

        optimizer_kws: dict, optional
            Keyword arguments to be passed to the ``get_sensors`` method of the optimizer.
        """

        # Fit basis functions to data
        # TODO: base class should have a _fit_basis method
        if prefit_basis:
            check_is_fitted(self.basis, "basis_matrix_")
        else:
            x = validate_input(x)

            with warnings.catch_warnings():
                action = "ignore" if quiet else "default"
                warnings.filterwarnings(action, category=UserWarning)
                self.basis.fit(x)

        # Get matrix representation of basis
        self.basis_matrix_ = self.basis.matrix_representation(
            n_basis_modes=self.n_basis_modes
        )

        # Check that n_sensors doesn't exceed dimension of basis vectors
        self._validate_n_sensors()

        # Find weight vector
        # TODO: should this be projection of x onto basis?
        self.optimizer.fit(self.basis_matrix_.T, y)

        self.w_ = self.optimizer.coef_

        # TODO: cvx routine to learn sensors
        s = CVX_routine(...)

        # Get sensor locations from s

    def predict(self, x):
        """
        Predict classes for given measurements.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_sensors)
            Measurements from which to form prediction.
            The measurements should be taken at the sensor locations specified by
            ``self.get_ranked_sensors()``.

        Returns
        -------
        y: numpy array, shape (n_samples,)
            Predicted classes.
        """

        return np.zeros(x.shape[0])
