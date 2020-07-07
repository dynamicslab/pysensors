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
    n_components : int, optional (default 10)
        Number of basis modes to retain. Cannot be larger than
        the number of features `n_features`, or the number of examples
        `n_examples`.

    algorithm: string, optional (default "randomized")
        SVD solver to use. Either “arpack” for the ARPACK wrapper
        in SciPy (scipy.sparse.linalg.svds), or “randomized” for the
        randomized algorithm due to Halko (2009).

    kwargs : dict, optional
        Keyword arguments to be passed to the TruncatedSVD constructor

    Attributes
    ----------
    basis_matrix_ : numpy ndarray, shape (n_features, n_components)
        The top n_components left singular vectors of the training data.

    References
    -----
    https://scikit-learn.org/stable/modules/generated/\
    sklearn.decomposition.TruncatedSVD.html

    Finding structure with randomness: Stochastic algorithms for
    constructing approximate matrix decompositions Halko, et al.,
    2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf
    """

    def __init__(self, n_components=10, **kwargs):
        super(POD, self).__init__(n_components=n_components, **kwargs)

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
        self.basis_matrix_ = super(POD, self).fit(X)

    def matrix_representation(self, copy=False):
        """
        Get the matrix representation of the operator.

        Parameters
        ----------
        copy : boolean, optional (default False)
            Whether to return a copy of the basis matrix.
        """
        check_is_fitted(self, "components_")
        # Note: the TruncatedSVD object returns components as rows, so we
        # take a transpose here.
        return self.components_.T.copy() if copy else self.components_.T
