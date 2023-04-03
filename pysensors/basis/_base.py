from abc import ABC
from abc import abstractmethod

from sklearn.utils.validation import check_is_fitted


class InvertibleBasis(ABC):
    """
    An abstract class ensuring subclasses implement a
    ``matrix_inverse`` method.
    """

    @abstractmethod
    def matrix_inverse(self, n_basis_modes=None, **kwargs):
        raise NotImplementedError("This method has not been implemented")


class MatrixMixin:
    """
    Mixin class for generating matrix representations of a basis.
    """

    def matrix_representation(self, n_basis_modes=None, copy=False):
        """
        Get the matrix representation of the training data in the basis.
        Note that in general this matrix is not the matrix whose column vectors
        are the basis modes.

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
        n_basis_modes = self._validate_input(n_basis_modes)

        if copy:
            return self.basis_matrix_[:, :n_basis_modes].copy()
        else:
            return self.basis_matrix_[:, :n_basis_modes]

    def _validate_input(self, n_basis_modes):
        """
        Ensure ``n_basis_modes`` does not exceed the maximum number possible.
        """
        check_is_fitted(self, "basis_matrix_")

        if n_basis_modes is None:
            n_basis_modes = self.n_basis_modes
        elif n_basis_modes > self.n_basis_modes:
            raise ValueError(
                f"Requested number of modes {n_basis_modes} exceeds"
                f" number available: {self.n_basis_modes}"
            )
        return n_basis_modes
