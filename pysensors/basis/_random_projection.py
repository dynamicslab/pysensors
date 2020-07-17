"""
Random projections basis class.

Project data onto random features.
"""
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils.validation import check_is_fitted


class RandomProjection(GaussianRandomProjection):
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

    def matrix_representation(self, copy=False):
        """
        Get the matrix representation of the operator.

        Parameters
        ----------
        copy : boolean, optional (default False)
            Whether to return a copy of the basis matrix.
        """
        check_is_fitted(self, "basis_matrix_")
        return self.basis_matrix_.copy() if copy else self.basis_matrix_
