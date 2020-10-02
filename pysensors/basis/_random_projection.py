"""
Random projections basis class.

Project data onto random features.
"""
from numpy.linalg import pinv
from sklearn.random_projection import GaussianRandomProjection

from ._base import InvertibleBasis
from ._base import MatrixMixin


class RandomProjection(GaussianRandomProjection, InvertibleBasis, MatrixMixin):
    """
    Generate a basis based on Gaussian random projection.

    Wrapper for sklearn.random_projection.GaussianRandomProjection.

    Parameters
    ----------
    n_basis_modes : int or "auto", optional (default 10)
        Dimensionality of the target projection space (number of modes).

        `n_basis_modes` can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the eps parameter.

    eps: positive float, optional (default 0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_basis_modes is set to ‘auto’.

        Smaller values lead to better embedding and higher number of
        dimensions (n_basis_modes) in the target projection space.

    random_state: int, RandomState instance or None, optional (default=None)
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time. Pass an int for reproducible output
        across multiple function calls.

    Attributes
    ----------
    basis_matrix_ : numpy ndarray, shape (n_features, n_basis_modes)
        Random projections of training data.

    References
    -----
    https://scikit-learn.org/stable/modules/generated/\
    sklearn.random_projection.GaussianRandomProjection.html

    """

    def __init__(self, n_basis_modes=10, eps=0.1, random_state=None):
        if (
            isinstance(n_basis_modes, int) and n_basis_modes > 0
        ) or n_basis_modes == "auto":
            super(RandomProjection, self).__init__(
                n_components=n_basis_modes, eps=eps, random_state=random_state
            )
            self.n_basis_modes = n_basis_modes
        else:
            raise ValueError("n_basis_modes must be a positive int or 'auto'")

    def fit(self, X):
        """
        X : array-like, shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : instance
        """

        super(RandomProjection, self).fit(X.T)
        self.basis_matrix_ = super(RandomProjection, self).transform(X.T)
        return self

    def matrix_inverse(self, n_basis_modes=None, **kwargs):
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

        return pinv(self.basis_matrix_[:, :n_basis_modes], **kwargs)

    @property
    def n_basis_modes(self):
        """Number of basis modes."""
        return self._n_basis_modes

    @n_basis_modes.setter
    def n_basis_modes(self, n_basis_modes):
        self._n_basis_modes = n_basis_modes
        self.n_components = n_basis_modes
