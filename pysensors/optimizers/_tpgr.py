import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class TPGR(BaseEstimator):
    """
    Two-Point Greedy Algorithm for Sensor Selection.

    See the following reference for more information

        Klishin, Andrei A., et. al.
        Data-Induced Interactions of Sparse Sensors. 2023.
        arXiv:2307.11838 [cond-mat.stat-mech]

    Parameters
    ----------
    n_sensors : int
        The number of sensors to select.

    prior: str or np.ndarray shape (n_basis_modes,), optional (default='decreasing')
        Prior Covariance Vector, typically a scaled identity vector or a vector
        containing normalized singular values. If 'decreasing', normalized singular
        values are used.

    noise: float (default None)
        Magnitude of the gaussian uncorrelated sensor measurement noise.

    Attributes
    ----------
    sensors_ : list of int
        Indices of the selected sensors (rows from the basis matrix).

    """

    def __init__(self, n_sensors, prior="decreasing", noise=None):
        self.n_sensors = n_sensors
        self.noise = noise
        self.sensors_ = None
        self.prior = prior

    def fit(self, basis_matrix, singular_values):
        """
        Parameters
        ----------
        basis_matrix: np.ndarray, shape (n_features, n_basis_modes)
            Matrix whose columns are the basis vectors in which to
            represent the measurement data.

        singular_values : np.ndarray, shape (n_basis_modes,)
            Normalized singular values to be used if `prior="decreasing"`.

        Returns
        -------
        self: a fitted :class:`pysensors.optimizers.TPGR` instance
        """
        if isinstance(self.prior, str) and self.prior == "decreasing":
            computed_prior = singular_values
        elif isinstance(self.prior, np.ndarray):
            if self.prior.ndim != 1:
                raise ValueError("prior must be a 1D array.")
            if self.prior.shape[0] != basis_matrix.shape[1]:
                raise ValueError(
                    f"prior must be of shape {(basis_matrix.shape[1],)},"
                    f" but got {self.prior.shape[0]}."
                )
            computed_prior = self.prior
        else:
            raise ValueError(
                "Invalid prior: must be 'decreasing' or a 1D "
                "ndarray of appropriate length."
            )
        if self.noise is None:
            warnings.warn(
                "noise is None. noise will be set to the average of the computed prior."
            )
            self.noise = computed_prior.mean()
        G = basis_matrix @ np.diag(computed_prior)
        n = G.shape[0]
        if self.n_sensors > G.shape[0]:
            raise ValueError("n_sensors cannot exceed the number of available sensors.")
        mask = np.ones(n, dtype=bool)
        one_pt_energies = self._one_pt_energy(G)
        i = np.argmin(one_pt_energies)
        self.sensors_ = [i]
        mask[i] = False
        G_selected = G[[i], :]
        while G_selected.shape[0] < self.n_sensors:
            G_remaining = G[mask]
            q = np.argmin(
                self._one_pt_energy(G_remaining)
                + 2 * self._two_pt_energy(G_selected, G_remaining)
            )
            remaining_indices = np.where(mask)[0]
            selected_index = remaining_indices[q]
            self.sensors_.append(selected_index)
            mask[selected_index] = False
            G_selected = np.vstack(
                (G_selected, G[selected_index : selected_index + 1, :])
            )
        return self

    def _one_pt_energy(self, G):
        """
        Compute the one-pt energy of the sensors

        Parameters
        ----------
        G : np.ndarray, shape (n_features, n_basis_modes)
            Basis matrix weighted by the prior.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        return -np.log(1 + np.einsum("ij,ij->i", G, G) / self.noise**2)

    def _two_pt_energy(self, G_selected, G_remaining):
        """
        Compute the two-pt energy interations of the selected
        sensors with the remaining sensors

        Parameters
        ----------
        G_selected : np.ndarray, shape (k, n_basis_modes)
            Matrix of currently selected k sensors.

        G_remaining : np.ndarray, shape (n_features - k, n_basis_modes)
            Matrix of currently remaining sensors.

        Returns
        -------
        np.ndarray, shape (n_features - k,)
        """
        J = 0.5 * np.sum(
            ((G_remaining @ G_selected.T) ** 2)
            / (
                np.outer(
                    1 + (np.sum(G_remaining**2, axis=1)) / self.noise**2,
                    1 + (np.sum(G_selected**2, axis=1)) / self.noise**2,
                )
                * self.noise**4
            ),
            axis=1,
        )
        return J

    def get_sensors(self):
        check_is_fitted(self, "sensors_")
        return self.sensors_
