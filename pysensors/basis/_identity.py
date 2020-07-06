"""
Identity basis class.

This is essentially a dummy basis which just uses raw, unaltered features.
"""
from sklearn.base import TransformerMixin
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
        return self.basis_matrix_
