import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class TPGR(BaseEstimator):
    """
    2-Point Greedy Algorithm for Sensor Selection.

    See the following reference for more information

        Klishin, Andrei A., et. al.
        Data-Induced Interactions of Sparse Sensors. 2023.
        arXiv:2307.11838 [cond-mat.stat-mech]

    """

    def __init__(self, n_sensors=None, noise=None, prior="decreasing"):
        self.n_sensors = n_sensors
        self.noise = noise
        self.sensors_ = None
        self.prior = prior

    def fit(self, basis_matrix, singular_values=None):
        if isinstance(self.prior, str) and self.prior == "decreasing":
            computed_prior = singular_values
        elif isinstance(self.prior, np.ndarray):
            if self.prior.ndim != 1:
                raise ValueError("prior must be a 1D array")
            if self.prior.shape[0] != basis_matrix.shape[1]:
                raise ValueError(
                    f"prior must be of shape {(basis_matrix.shape[1],)},"
                    f" but got {self.prior.shape[0]}"
                )
            computed_prior = self.prior
        if self.noise is None:
            warnings.warn(
                "noise is None. noise will be set to the " "average of the prior"
            )
            self.noise = computed_prior.mean()
        G = basis_matrix @ np.diag(computed_prior)
        self.G = G
        n = G.shape[0]
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
        Compute the 1-pt energy
        """
        return -np.log(1 + np.einsum("ij,ij->i", G, G) / self.noise**2)

    def _two_pt_energy(self, G_selected, G_remaining):
        """
        Compute the 2-pt energy
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
