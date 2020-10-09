"""
SVD mode basis class.
"""
from sklearn.decomposition import TruncatedSVD

from ._base import InvertibleBasis
from ._base import MatrixMixin


class SVD(TruncatedSVD, InvertibleBasis, MatrixMixin):
    """
    Generate an SVD transformation which maps input features to
    SVD modes.

    Assumes the data has already been centered (to have mean 0).

    Parameters
    ----------
    n_basis_modes : int, optional (default 10)
        Number of basis modes to retain. Cannot be larger than
        the number of features ``n_features``, or the number of examples
        ``n_examples``.

    algorithm: string, optional (default "randomized")
        SVD solver to use. Either “arpack” for the ARPACK wrapper
        in SciPy (scipy.sparse.linalg.svds), or “randomized” for the
        randomized algorithm due to Halko (2009).

    kwargs : dict, optional
        Keyword arguments to be passed to the TruncatedSVD constructor

    Attributes
    ----------
    basis_matrix_ : numpy ndarray, shape (n_features, n_basis_modes)
        The top n_basis_modes left singular vectors of the training data.

    References
    -----
    https://scikit-learn.org/stable/modules/generated/\
    sklearn.decomposition.TruncatedSVD.html

    Finding structure with randomness: Stochastic algorithms for
    constructing approximate matrix decompositions Halko, et al.,
    2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf
    """

    def __init__(self, n_basis_modes=10, **kwargs):
        if isinstance(n_basis_modes, int) and n_basis_modes > 0:
            super(SVD, self).__init__(n_components=n_basis_modes, **kwargs)
            self._n_basis_modes = n_basis_modes
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
        self.basis_matrix_ = super(SVD, self).fit(X).components_.T
        return self

    def matrix_inverse(self, n_basis_modes=None):
        """
        Get the inverse matrix mapping from measurement space to
        coordinates with respect to the basis.

        Note that this is not the inverse of the matrix returned by
        ``self.matrix_representation``. It is the (psuedo) inverse of
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
