"""
Identity basis class.

This is essentially a dummy basis which just uses raw, unaltered features.
"""
from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from ._base import MatrixMixin


class Identity(BaseEstimator, MatrixMixin):
    """
    Generate an identity transformation which maps all input features to
    themselves.

    Parameters
    ----------
    n_basis_modes: int, optional (default None)
        Number of basis modes to retain. If None, all are included
        in the basis.

    Attributes
    ----------
    basis_matrix_ : numpy ndarray, shape (n_features, n_samples)
        The transpose of the input data.
    """

    def __init__(self, n_basis_modes=None):
        if n_basis_modes is None:
            self.n_basis_modes = None
        elif isinstance(n_basis_modes, int) and n_basis_modes > 0:
            self.n_basis_modes = n_basis_modes
        else:
            raise ValueError("n_basis_modes must be a positive integer.")

    def fit(self, X):
        """
        Memorize the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : instance
        """

        # Note that we take a transpose here, so columns correspond to examples
        if self.n_basis_modes is None:
            self.basis_matrix_ = check_array(X).T.copy()
            self.n_basis_modes = self.basis_matrix_.shape[1]
        else:
            if self.n_basis_modes > X.shape[0]:
                raise ValueError(
                    "X needs at least n_basis_modes ({}) examples/rows".format(
                        self.n_basis_modes
                    )
                )

            self.basis_matrix_ = check_array(X)[: self.n_basis_modes, :].T.copy()

            if self.n_basis_modes < X.shape[0]:
                warn(f"Only the first {self.n_basis_modes} examples were retained.")
        return self
