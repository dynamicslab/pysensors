import numpy as np

from ._qr import QR


class CCQR(QR):
    """
    Greedy cost-constrained QR optimizer for sensor selection.

    This method is based on the following work:

        Clark, Emily, et al.
        "Greedy sensor placement with cost constraints."
        IEEE Sensors Journal 19.7 (2018): 2642-2656.
    """

    def __init__(self, sensor_costs=None):
        """
        Greedy cost-constrained QR optimizer for sensor selection.
        This algorithm augments the pivot selection criteria used in the
        QR algorithm (with householder reflectors) to take into account
        costs associated with each sensors. It is similar to the
        :class:`pysensors.optimizers.QR` algorithm in that it returns an array
        of sensor locations ranked by importance, but with a definition of
        importance that takes sensor costs into account.

        Parameters
        ----------
        sensor_costs: np.ndarray, shape [n_features,], optional (default None)
            Costs (weights) associated with each sensor.
            Positive values will encourage sensors to be avoided and
            negative values will cause them to be preferred.
            If None, costs will all be set to zero.

        Attributes
        ----------
        pivots_ : np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        """
        super(CCQR, self).__init__()
        if sensor_costs is not None and np.ndim(sensor_costs) != 1:
            raise ValueError(
                "sensor_costs must be a 1D array, "
                f"but a {np.ndim(sensor_costs)}D array was given"
            )
        self.sensor_costs = sensor_costs

    def fit(
        self,
        basis_matrix,
    ):
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
        self: a fitted :class:`pysensors.optimizers.CCQR` instance
        """

        n, m = basis_matrix.shape  # We transpose basis_matrix below

        if self.sensor_costs is None:
            self.sensor_costs = np.zeros(n)

        if len(self.sensor_costs) != n:
            raise ValueError(
                f"Dimension of sensor_costs ({len(self.sensor_costs)}) "
                f"does not match number of sensors in data ({n})"
            )

        # Initialize helper variables
        R = basis_matrix.conj().T.copy()
        p = np.arange(n)
        k = min(m, n)

        for j in range(k):
            u, i_piv = qr_reflector(R[j:, j:], self.sensor_costs[p[j:]])
            # Track column pivots
            i_piv += j
            p[[j, i_piv]] = p[[i_piv, j]]
            # Switch columns
            R[:, [j, i_piv]] = R[:, [i_piv, j]]
            # Apply reflector
            R[j:, j:] -= np.outer(u, np.dot(u, R[j:, j:]))
            R[j + 1 :, j] = 0

        self.pivots_ = p

        return self


def qr_reflector(r, costs):
    """
    Get the best (Householder) reflector with column pivoting and
    a cost function.

    The pivoting is biased by a cost function, i.e.
    the pivot is chosen as the argmax of :code:`norm(r[:, i]) - costs[i]`,
    whereas normally it would be chosen as the argmax of :code:`norm(r[:, i])`.

    Parameters
    ----------
    r: np.ndarray, shape [n_features, n_examples]
        Sub-array for which the pivot and reflector are to be found

    costs: np.ndarray, shape [n_examples,]
        Costs for each column (sensor location) in r

    Returns
    -------
    u: np.ndarray, shape [n_features,]
        Householder reflector.

    i_piv: nonnegative integer
        Index of the pivot.
    """

    # Norm of each column
    dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))

    # Choose pivot
    i_piv = np.argmax(dlens - costs)

    dlen = dlens[i_piv]

    if dlen > 0:
        u = r[:, i_piv] / dlen
        u[0] += np.sign(u[0]) + (u[0] == 0)
        u /= np.sqrt(abs(u[0]))
    else:
        u = r[:, i_piv]
        u[0] = np.sqrt(2)

    return u, i_piv
