import numpy as np

from ._qr import QR


class GQR(QR):
    """
    General QR optimizer for sensor selection.
    Ranks sensors in descending order of "importance" based on
    reconstruction performance. This is an extension that requires a more intrusive
    access to the QR optimizer to facilitate a more adaptive optimization. This is a generalized version of cost constraints
    in the sense that users can allow n consttrained sensors in the constrained area.
    if n = 0 this converges to the CCQR results.
    @ authors: Niharika Karnik (@nkarnik2999), Mohammad Abdo (@Jimmy-INL), and Krithika Manohar (@kmanohar)
    """
    def __init__(self,idx_constrained,n_sensors,const_sensors):
        """
        Attributes
        ----------
        pivots_ : np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        """
        self.pivots_ = None
        self.constrainedIndices = idx_constrained
        self.nSensors = n_sensors
        self.nConstrainedSensors = const_sensors

    def fit(
        self,
        basis_matrix
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

        n_features, n_samples = basis_matrix.shape  # We transpose basis_matrix below
        max_const_sensors = len(self.constrainedIndices)

        ## Assertions and checks:
        if self.nSensors > n_features - max_const_sensors + self.nConstrainedSensors:
            raise IOError ("n_sensors cannot be larger than n_features - all possible locations in the constrained area + allowed constrained sensors")
        if self.nSensors > n_samples + self.nConstrainedSensors:
            raise IOError ("Currently n_sensors should be less than number of samples + number of constrained sensors,\
                           got: n_sensors = {}, n_samples + const_sensors = {} + {} = {}".format(n_sensors,n_samples,const_sensors,n_samples+const_sensors))

        # Initialize helper variables
        R = basis_matrix.conj().T.copy()
        #print(R.shape)
        p = np.arange(n_features)
        #print(p)
        k = min(n_samples, n_features)


        for j in range(k):
            r = R[j:, j:]
            # Norm of each column
            dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))

            # if j < const_sensors:
            dlens_updated = f_region(self.constrainedIndices,dlens,p,j, self.nConstrainedSensors)
            # else:
                # dlens_updated = dlens
            # Choose pivot
            i_piv = np.argmax(dlens_updated)
            #print(i_piv)


            dlen = dlens_updated[i_piv]

            if dlen > 0:
                u = r[:, i_piv] / dlen
                u[0] += np.sign(u[0]) + (u[0] == 0)
                u /= np.sqrt(abs(u[0]))
            else:
                u = r[:, i_piv]
                u[0] = np.sqrt(2)

            # Track column pivots
            i_piv += j # true permutation index is i_piv shifted by the iteration counter j
            # print(i_piv) # Niharika's debugging line
            p[[j, i_piv]] = p[[i_piv, j]]
            # print(p)


            # Switch columns
            R[:, [j, i_piv]] = R[:, [i_piv, j]]

            # Apply reflector
            R[j:, j:] -= np.outer(u, np.dot(u, R[j:, j:]))
            R[j + 1 :, j] = 0

        self.pivots_ = p


        return self

## TODO: why not a part of the class?
#function for mapping sensor locations with constraints
def f_region(lin_idx, dlens, piv, j, const_sensors):
    #num_sensors should be fixed for each custom constraint (for now)
    #num_sensors must be <= size of constraint region
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
        ----------
        lin_idx: np.ndarray, shape [No. of constrained locations]
            Array which contains the constrained locations mapped on the grid.
        dlens: np.ndarray, shape [Variable based on j]
            Array which contains the norm of columns of basis matrix.
         num_sensors: int,
            Number of sensors to be placed in the constrained area.
        j: int,
            Iterative variable in the QR algorithm.

        Returns
        -------
        dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if j < const_sensors: # force sensors into constraint region
        #idx = np.arange(dlens.shape[0])
        #dlens[np.delete(idx, lin_idx)] = 0

        didx = np.isin(piv[j:],lin_idx,invert=True)
        dlens[didx] = 0
    else:
        didx = np.isin(piv[j:],lin_idx,invert=False)
        dlens[didx] = 0
    return dlens

def getConstraindSensorsIndices(xmin, xmax,ymin,ymax, all_sensors):
    n_features = len(all_sensors)
    imageSize = int(np.sqrt(n_features))
    a = np.unravel_index(all_sensors, (imageSize,imageSize))
    constrained_sensorsx = []
    constrained_sensorsy = []
    for i in range(n_features):
        if (a[0][i] > xmin and a[0][i] < xmax) and (a[1][i] > ymin and a[1][i] < ymax):  # x<10 and y>40
            constrained_sensorsx.append(a[0][i])
            constrained_sensorsy.append(a[1][i])

    constrained_sensorsx = np.array(constrained_sensorsx)
    constrained_sensorsy = np.array(constrained_sensorsy)
    constrained_sensors_array = np.stack((constrained_sensorsy, constrained_sensorsx), axis=1)
    constrained_sensors_tuple = np.transpose(constrained_sensors_array)
    if len(constrained_sensorsx) == 0:
        idx_constrained = []
    else:
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (imageSize,imageSize))
    return idx_constrained

def boxConstraints(position,lowerBound,upperBound,):
    for i,xi in enumerate(position):
        f1 = position[i] - lowerBound[i]
        f2 = upperBound[i] - position [i]
    return +1 if (f1 and f2 > 0) else -1

def functionalConstraint(position, func_response,func_input, freeTerm):
    g = func_response + func_input + freeTerm
    return g


if __name__ == '__main__':
    pass
