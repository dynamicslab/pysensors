import numpy as np

class CCQR:
    """
    Greedy cost-constrained QR optimizer for sensor selection.
    """

    def __init__(self, cost_function):
        if np.ndim(cost_function) != 1:
            raise ValueError(
                "cost_function must be a 1D array, "
                f"but a {np.ndim(cost_function)}D array was given"
            )
        self.cost_function = cost_function

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
        cost_function: Cost function on sensor location, shape [n_features, 1]
        """

        #TODO: Translate code from Matlab to Python

        basis_matrixT = basis_matrix.conj().T
        m, n = basis_matrixT.shape

        if len(cost_function) != n:
            raise ValueError(
                    f"vector of costs has wrong dimensions"
                )

        # initialize
        QH = np.zeros([m,n])
        R = basis_matrixT
        self.p = list(range(n))

        k = min(m,n)

        for j in range(k):
            u, ipiv = qrpc_reflector(np.array(R[j:m,j:n]),np.array(cost_function[p[j:n]]))
            # track column pivots
            ipiv = ipiv+j
            itemp = self.p[j]
            self.p[j] = self.p[ipiv]
            self.p[ipiv] = itemp
            # switch columns
            temp = R[:,j]
            R[:,j] = R[:,ipiv]
            R[:,ipiv] = temp
            # apply reflector
            QH[j:m,j] = u
            s = np.dot(u.T,R[j:m,j:n])
            R[j:m,j:n] = R[j:m,j:n] - np.outer(u,s)
            R[j+1:m,j] = 0

        return self.p


def qrpc_reflector(r,cost_function):
    """
    QRPC_REFLECTOR best reflector with column pivoting and a cost function

    This function generates a Householder reflector
    The pivoting is biased by a cost function, i.e.
    the pivot is chosen as the argmax of norm(r(:,i))-cost_function(i)
    """

    # size of each column
    r2 = np.multiply(r,r.conj())
    sum_r2 = r2.sum(axis=0)
    dlens = np.sqrt(sum_r2)

    # choose pivot
    ipiv = np.argmax(dlens-cost_function)

    dlen = dlens[ipiv]

    if dlen > 0.0:
        u = r[:,ipiv]/dlen
        u[0] = u[0] + np.sign(u[0]) + (u[0] == 0)
        u = u/np.sqrt(abs(u[0]))
    else:
        u = r[:,ipiv]
        u[0] = np.sqrt(2.0)

    return u, ipiv
