import numpy as np

from ._qr import QR


class CCQR(QR):
    """
    Greedy cost-constrained QR optimizer for sensor selection.
    """

    def __init__(self, sensor_costs):
        """
        TODO

        Parameters
        ----------
        sensor_costs: np.ndarray, shape (n_features,)
            Costs (weights) associated with each sensor.

        """
        super(CCQR, self).__init__()
        if np.ndim(sensor_costs) != 1:
            raise ValueError(
                "sensor_costs must be a 1D array, "
                f"but a {np.ndim(sensor_costs)}D array was given"
            )
        self.sensor_costs = sensor_costs

    def get_sensors(
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
        """

        n, m = basis_matrix.shape  # We transpose basis_matrix below

        if len(self.sensor_costs) != n:
            raise ValueError(
                f"Dimension of sensor_costs ({len(self.sensor_costs)}) "
                f"does not match number of sensors in data ({n})"
            )

        # Use scipy QR if no sensor costs for efficiency
        # if np.count_nonzero(self.sensor_costs) == 0:
        #     return super(CCQR, self).get_sensors(basis_matrix)

        # Initialize helper variables
        R = basis_matrix.conj().T.copy()
        p = list(range(n))

        k = min(m, n)

        for j in range(k):
            u, i_piv = qrpc_reflector(R[j:, j:], self.sensor_costs[p[j:]])
            # Track column pivots
            i_piv += j
            p[j], p[i_piv] = p[i_piv], p[j]
            # Switch columns
            R[:, [j, i_piv]] = R[:, [i_piv, j]]
            # Apply reflector
            R[j:, j:] -= np.outer(u, np.dot(u, R[j:, j:]))
            R[j + 1 :, j] = 0

        self.pivots_ = p
        return p


def qrpc_reflector(r, cost_function):
    """
    QRPC_REFLECTOR best reflector with column pivoting and a cost function

    This function generates a Householder reflector
    The pivoting is biased by a cost function, i.e.
    the pivot is chosen as the argmax of norm(r(:, i)) - cost_function(i)
    """

    # Norm of each column
    dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))

    # Choose pivot
    i_piv = np.argmax(dlens - cost_function)

    dlen = dlens[i_piv]

    if dlen > 0:
        u = r[:, i_piv] / dlen
        u[0] += np.sign(u[0]) + (u[0] == 0)
        u /= np.sqrt(abs(u[0]))
    else:
        u = r[:, i_piv]
        u[0] = np.sqrt(2)

    return u, i_piv
