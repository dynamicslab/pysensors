"""
POD mode basis class.
"""
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted


class POD(TruncatedSVD):
    """
    Generate a POD transformation which maps input features to
    POD modes.

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
            super(POD, self).__init__(n_components=n_basis_modes, **kwargs)
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
        self.basis_matrix_ = super(POD, self).fit(X).components_.T
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
        # Note: the TruncatedSVD object returns components as rows, so we
        # take a transpose here.
        return self.basis_matrix_.copy() if copy else self.basis_matrix_
