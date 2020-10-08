from scipy.linalg import qr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class QR(BaseEstimator):
    """
    Greedy QR optimizer for sensor selection.
    Ranks sensors in descending order of "importance" by applying
    the QR algorithm and extracting pivot indices.

    See the following reference for more information

        Manohar, Krithika, et al.
        "Data-driven sparse sensor placement for reconstruction:
        Demonstrating the benefits of exploiting known patterns."
        IEEE Control Systems Magazine 38.3 (2018): 63-86.
    """

    def __init__(self):
        """
        Attributes
        ----------
        pivots_ : np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        """
        self.pivots_ = None

    def fit(self, basis_matrix, **optimizer_kws):
        """
        Parameters
        ----------
        basis_matrix: np.ndarray, shape [n_features, n_samples]
            Matrix whose columns are the basis vectors in which to
            represent the measurement data.

        optimizer_kws: dictionary, optional
            Keyword arguments to be passed to the qr method.

        Returns
        -------

        """

        # TODO: implement checks on basis_matrix
        _, _, self.pivots_ = qr(basis_matrix.conj().T, pivoting=True, **optimizer_kws)

        return self

    def get_sensors(self):
        """
        Get ranked array of sensors.

        Returns
        -------
        sensors: np.ndarray, shape [n_features,]
            Array of sensors ranked in descending order of importance.
            Note that if n_features exceeds n_samples, then only the first
            n_samples entries of sensors are guaranteed to be in ranked order.
        """
        check_is_fitted(self, "pivots_")
        return self.pivots_
