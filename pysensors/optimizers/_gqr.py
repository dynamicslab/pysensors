import numpy as np
import pysensors

from pysensors.optimizers._qr import QR

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pysensors as ps
from matplotlib.patches import Circle
from pysensors.utils._norm_calc import returnInstance as normCalcReturnInstance

class GQR(QR):
    """
    General QR optimizer for sensor selection.
    Ranks sensors in descending order of "importance" based on
    reconstruction accuracy. This is an extension that requires a more intrusive
    access to the QR optimizer to facilitate a more adaptive optimization. This is a generalized version of cost constraints
    in the sense that users can allow `n_const_sensors` in the constrained area.
    if n = 0 this converges to the CCQR results. and if no constrained region it should converge to the results from QR optimizer.

    See the following reference for more information
        Manohar, Krithika, et al.
        "Data-driven sparse sensor placement for reconstruction:
        Demonstrating the benefits of exploiting known patterns."
        IEEE Control Systems Magazine 38.3 (2018): 63-86.

    @ authors: Niharika Karnik (@nkarnik2999), Mohammad Abdo (@Jimmy-INL), and Krithika Manohar (@kmanohar)
    """
    def __init__(self):
        """
        Attributes
        ----------
        pivots_ : np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        idx_constrained : np.ndarray, shape [No. of constrained locations]
            Column Indices of the sensors in the constrained locations.
        n_sensors : integer,
            Total number of sensors
        n_const_sensors : integer,
            Total number of sensors required by the user in the constrained region.
        all_sensors : np.ndarray, shape [n_features]
            Optimally placed list of sensors obtained from QR pivoting algorithm.
        constraint_option : string,
            max_n_const_sensors : The number of sensors in the constrained region should be less than or equal to n_const_sensors.
            exact_n_const_sensors : The number of sensors in the constrained region should be exactly equal to n_const_sensors.
        """
        self.pivots_ = None
        self.idx_constrained = []
        self.n_sensors = None
        self.n_const_sensors = 0
        self.all_sensors = []
        self.constraint_option = ''
        self.nx = None
        self.ny = None
        self.r = 1

    def fit(self,basis_matrix,**optimizer_kws):
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
        self: a fitted :class:`pysensors.optimizers.GQR` instance
        """
        [setattr(self,name,optimizer_kws.get(name,getattr(self,name))) for name in optimizer_kws.keys()]
        self._norm_calc_Instance = normCalcReturnInstance(self, self.constraint_option)
        n_features, n_samples = basis_matrix.shape  # We transpose basis_matrix below
        max_const_sensors = len(self.idx_constrained) # Maximum number of sensors allowed in the constrained region

        ## Assertions and checks:
        # if self.n_sensors > n_features - max_const_sensors + self.nConstrainedSensors:
        #     raise IOError ("n_sensors cannot be larger than n_features - all possible locations in the constrained area + allowed constrained sensors")
        # if self.n_sensors > n_samples + self.nConstrainedSensors: ## Handling zero constraint?
        #     raise IOError ("Currently n_sensors should be less than min(number of samples, number of modes) + number of constrained sensors,\
        #                    got: n_sensors = {}, n_samples + const_sensors = {} + {} = {}".format(self.n_sensors,n_samples,self.nConstrainedSensors,n_samples+self.nConstrainedSensors))

        # Initialize helper variables
        R = basis_matrix.conj().T.copy()
        p = np.arange(n_features)
        k = min(n_samples, n_features)


        for j in range(k):
            r = R[j:, j:]

            # Norm of each column
            if j == 0:
                dlens_old = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))
            else:
                dlens_old = dlens
            dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))
            dlens_updated = self._norm_calc_Instance(self.idx_constrained, dlens, p, j, self.n_const_sensors, dlens_old=dlens_old, all_sensors=self.all_sensors, n_sensors=self.n_sensors, nx=self.nx, ny=self.ny, r=self.r)
            # i_piv = np.argmax(dlens_updated)
            i_piv = np.where(dlens_updated==dlens_updated.max())[0][0]
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
            p[[j, i_piv]] = p[[i_piv, j]]

            # Switch columns
            R[:, [j, i_piv]] = R[:, [i_piv, j]]

            # Apply reflector
            R[j:, j:] -= np.outer(u, np.dot(u, R[j:, j:]))
            R[j + 1 :, j] = 0
        self.pivots_ = p
        return self