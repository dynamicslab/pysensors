"""
Identity basis class.

This is essentially a dummy basis which just uses raw, unaltered features.
"""

from numpy import identity
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class Identity(TransformerMixin):
    """
    Generate an identity transformation which maps all input features to
    themselves.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is equal to the number of input features.
    """

    def __init__(self):
        self.n_input_features_ = None

    def fit(self, X, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """

        # Note that we take a transpose here, so columns correspond to examples
        self.basis_matrix_ = check_array(X).T.copy()
        return self

    def matrix_representation(self):
        """Get the matrix representation of the operator.
        """
        check_is_fitted(self, "basis_matrix_")
        return identity(self.basis_matrix_)


    """
    I think we can get rid of these functions
    """

    def transform(self, X):
        """Perform identity transformation (return a copy of the input).
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        X : np.ndarray, shape [n_samples, n_features]
            The matrix of features, which is just a copy of the input data.
        """
        check_is_fitted(self)

        n_samples, n_features = check_array(X).shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        return X.copy()

    def inverse(self, X):
        """Perform the inverse of the identity mapping, which is just the
        identity mapping.

        Returns
        -------
        X : np.ndarray, shape [n_samples, n_features]
            The matrix of features, which is just a copy of the input data.

        """
        return self.transform(X)