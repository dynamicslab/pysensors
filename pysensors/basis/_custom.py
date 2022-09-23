"""
custom mode basis class.
"""
from ._base import InvertibleBasis
from ._base import MatrixMixin

class Custom(InvertibleBasis, MatrixMixin):
    """
<<<<<<< HEAD
    Use a custom transformation to maps input features to
=======
    Generate a custom transformation which maps input features to
>>>>>>> 1ccd23fafed0ca39dac6cde204efa228d133df1c
    custom modes.

    Assumes the data has already been centered (to have mean 0).

    Parameters
    ----------
    n_basis_modes : int, optional (default 10)
        Number of basis modes to retain. Cannot be larger than
        the number of features ``n_features``, or the number of examples
        ``n_examples``.
    U: The custom basis matrix

    Attributes
    ----------
    basis_matrix_ : numpy ndarray, shape (n_features, n_basis_modes)
        The top n_basis_modes left singular vectors of the training data.

    """

    def __init__(self, U, n_basis_modes=10, **kwargs):
        if isinstance(n_basis_modes, int) and n_basis_modes > 0:
            super(Custom, self).__init__()#n_components=n_basis_modes, **kwargs
            self._n_basis_modes = n_basis_modes
            self.custom_basis_ = U
        else:
            raise ValueError("n_basis_modes must be a positive integer.")

    def fit(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : instance
        """
        # self.basis_matrix_ = self.custom_basis_[:,: self.n_basis_modes] @ self.custom_basis_[:,: self.n_basis_modes].T @ X[: self.n_basis_modes, :].T.copy()
        # self.basis_matrix_ = self.custom_basis_ @ self.custom_basis_.T @ X[: self.n_basis_modes, :].T.copy()
        self.basis_matrix_ = self.custom_basis_[:,:self.n_basis_modes]
        # self.basis_matrix_ = (X @ self.custom_basis_[:,:self.n_basis_modes] @ self.custom_basis_[:,:self.n_basis_modes].T)[:self.n_basis_modes,:].T

        # self.basis_matrix_ = ((X @ self.custom_basis_).T)[:,:self.n_basis_modes]
        # self.basis_matrix_ = ((X @ self.custom_basis_ @ self.custom_basis_.T).T)[:,:self.n_basis_modes]
        return self

    def matrix_inverse(self, n_basis_modes=None):
        """
        Get the inverse matrix mapping from measurement space to
        coordinates with respect to the basis.

        Note that this is not the inverse of the matrix returned by
        ``self.matrix_representation``. It is the (pseudo) inverse of
        the matrix whose columns are the basis modes.

        Parameters
        ----------
        n_basis_modes : positive int, optional (default None)
            Number of basis modes to be used to compute inverse.

        Returns
        -------
        B : numpy ndarray, shape (n_basis_modes, n_features)
            The inverse matrix.
        """
        n_basis_modes = self._validate_input(n_basis_modes)

        return self.basis_matrix_[:, :n_basis_modes].T

    @property
    def n_basis_modes(self):
        """Number of basis modes."""
        return self._n_basis_modes

    @n_basis_modes.setter
    def n_basis_modes(self, n_basis_modes):
        self._n_basis_modes = n_basis_modes
        self.n_components = n_basis_modes
