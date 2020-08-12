from sklearn.utils.validation import check_is_fitted


class MatrixMixin:
    """
    Mixin class for generating matrix representation of a basis.
    """

    def matrix_representation(self, n_basis_modes=None, copy=False):
        """
        Get the matrix representation of the operator.

        Parameters
        ----------
        n_basis_modes: positive int, optional (default None)
            Number of basis modes of matrix representation to return.

        copy : boolean, optional (default False)
            Whether to return a copy of the basis matrix.

        Returns
        -------
        B : numpy array, shape (n_features, n_basis_modes)
            Matrix representation of the basis. Note that rows correspond to
            features and columns to basis modes.
        """
        check_is_fitted(self, "basis_matrix_")

        if n_basis_modes is None:
            return self.basis_matrix_.copy() if copy else self.basis_matrix_
        else:
            if n_basis_modes > self.n_basis_modes:
                raise ValueError(
                    f"Requested number of modes {n_basis_modes} exceeds"
                    f" number available: {self.n_basis_modes}"
                )
            if copy:
                return self.basis_matrix_[:, :n_basis_modes].copy()
            else:
                return self.basis_matrix_[:, :n_basis_modes]
