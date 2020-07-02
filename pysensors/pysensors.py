"""
SensorSelector object definition.
"""
from numpy import dot
from scipy.linalg import solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from pysensors.basis import Identity
from pysensors.optimizers import QR


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

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(self, basis=None, optimizer=None):
        if basis is None:
            basis = Identity()
        self.basis = basis
        if optimizer is None:
            self.optimizer = QR()
        self.optimizer = optimizer

    def fit(self, x, **optimizer_kws):
        """
        Fit the SensorSelector model, determining which sensors are relevant.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Training data.

        optimizer_kws: dictionary
            Keyword arguments to be fed into the `fit` method of the optimizer.
        """

        # TODO: some kind of preprocessing / quality control on x
        x = validate_input(x)

        # Fit basis functions to data (sometimes unnecessary, e.g FFT)
        basis.fit(x)

        # Get matrix representation of basis
        self.basis_matrix_ = basis.matrix_representation()

        # Apply the optimizer
        self.selected_sensors_ = self.optimizer.apply(
            x, self._basis_mat, **optimizer_kws
        )

    def predict(self, x, **solve_kws):
        """
        If x is a column vector, should behave fine.
        If x is a 2D array with rows corresponding to examples we'll need
        to transpose it before multiplying it with the basis matrix.
        """
        check_is_fitted(self, "selected_sensors_")
        x = validate_input(x, self.selected_sensors_)

        # For efficiency we may want to factor
        # self.basis_matrix_[self.selected_sensors_, :]
        # in case predict is called multiple times

        return dot(
            self.basis_matrix_,
            solve(self.basis_matrix_[self.selected_sensors_, :], x, **solve_kws),
        )
