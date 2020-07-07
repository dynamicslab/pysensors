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
    basis_matrix_ : numpy ndarray, shape (n_features, n_samples)
        The transpose of the input data.
    """

    def __init__(self):
        self.n_input_features_ = None

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
        self.basis_matrix_ = check_array(X).T.copy()
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
