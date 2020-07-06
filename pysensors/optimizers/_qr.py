from scipy.linalg import qr


class QR:
    """
    Greedy QR optimizer for sensor selection.
    """

    def __init__(self):
        self.pivots_ = None

    def get_sensors(self, basis_matrix, **optimizer_kws):
        """
        Parameters
        ----------
        basis_matrix: np.ndarray, shape [n_features, n_samples]
            Matrix whose columns are the basis vectors in which to
            represent the measurement data.

        optimizer_kws: dictionary, optional
            Keyword arguments to be passed to the qr method.
        """

        _, _, self.pivots_ = qr(basis_matrix.T, pivoting=True, **optimizer_kws)

        return self.pivots_
