import numpy as np
import pysensors

from pysensors.optimizers._qr import QR

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pysensors as ps
from matplotlib.patches import Circle


class GQR(QR):
    """
    General QR optimizer for sensor selection.
    Ranks sensors in descending order of "importance" based on
    reconstruction performance. This is an extension that requires a more intrusive
    access to the QR optimizer to facilitate a more adaptive optimization. This is a generalized version of cost constraints
    in the sense that users can allow n constrained sensors in the constrained area.
    if n = 0 this converges to the CCQR results.

    See the following reference for more information
        Manohar, Krithika, et al.
        "Data-driven sparse sensor placement for reconstruction:
        Demonstrating the benefits of exploiting known patterns."
        IEEE Control Systems Magazine 38.3 (2018): 63-86.

    @ authors: Niharika Karnik (@nkarnik2999), Mohammad Abdo (@Jimmy-INL), and Krithika Manohar (@kmanohar)
    """
    def __init__(self):#,idx_constrained,n_sensors,n_const_sensors,all_sensors,constraint_option,nx,ny,r
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
            Optimall placed list of sensors obtained from QR pivoting algorithm.
        constraint_option : string,
            max_n_const_sensors : The number of sensors in the constrained region should be less than or equal to n_const_sensors.
            exact_n_const_sensors : The number of sensors in the constrained region should be exactly equal to n_const_sensors.
        """
        self.pivots_ = None
        # self.optimality = None

        # self.constrainedIndices = idx_constrained
        # self.n_sensors = n_sensors
        # self.nConstrainedSensors = n_const_sensors
        # self.all_sensorloc = all_sensors
        # self.constraint_option = constraint_option
        # self._nx = nx
        # self._ny = ny
        # self._r = r

    def fit(self,basis_matrix=None,**optimizer_kws):
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
        self: a fitted :class:`pysensors.optimizers.QR` instance
        """
        if 'idx_constrained' in optimizer_kws.keys():
            self.constrainedIndices = optimizer_kws['idx_constrained']
        else:
            self.constrainedIndices = []
        if 'n_sensors' in optimizer_kws.keys():
            self.n_sensors = optimizer_kws['n_sensors']
        else:
            self.n_sensors = np.shape(basis_matrix)[0]
        if 'n_const_sensors' in optimizer_kws.keys():
            self.nConstrainedSensors = optimizer_kws['n_const_sensors']
        else:
            self.nConstrainedSensors = None
        if 'all_sensors' in optimizer_kws.keys():
            self.all_sensors = optimizer_kws['all_sensors']
        else:
            self.all_sensors = None
        if 'constraint_option' in optimizer_kws.keys():
            self.constraint_option = optimizer_kws['constraint_option']
        else:
            self.constraint_option = None
        if 'nx' in optimizer_kws.keys():
            self._nx = optimizer_kws['nx']
        else:
            self._nx = None
        if 'ny' in optimizer_kws.keys():
            self._ny = optimizer_kws['ny']
        else:
            self._ny = None
        if 'r' in optimizer_kws.keys():
            self._r = optimizer_kws['r']
        else:
            self._r = None

        n_features, n_samples = basis_matrix.shape  # We transpose basis_matrix below
        max_const_sensors = len(self.constrainedIndices) # Maximum number of sensors allowed in the constrained region

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
            dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))

            if self.constraint_option == "max_n_const_sensors" :
                dlens_updated = ps.utils._norm_calc.norm_calc_max_n_const_sensors(self.constrainedIndices,dlens,p,j, self.nConstrainedSensors,self.all_sensorloc,self.n_sensors)
                i_piv = np.argmax(dlens_updated)
                dlen = dlens_updated[i_piv]
            elif self.constraint_option == "exact_n_const_sensors" :
                dlens_updated = ps.utils._norm_calc.norm_calc_exact_n_const_sensors(self.constrainedIndices,dlens,p,j,self.nConstrainedSensors)
                i_piv = np.argmax(dlens_updated)
                dlen = dlens_updated[i_piv]
            elif self.constraint_option == "predetermined_end":
                dlens_updated = ps.utils._norm_calc.predetermined_norm_calc(self.constrainedIndices, dlens, p, j, self.nConstrainedSensors, self.n_sensors)
                i_piv = np.argmax(dlens_updated)
                dlen = dlens_updated[i_piv]
            elif self.constraint_option == "radii_constraints":

                if j == 0:
                    i_piv = np.argmax(dlens)
                    dlen = dlens[i_piv]
                    dlens_old = dlens
                else:

                    dlens_updated = ps.utils._norm_calc.f_radii_constraint(j,dlens,dlens_old,p,self._nx,self._ny,self._r,self.all_sensorloc, self.n_sensors) #( self.radius,self._nx,self._ny,self.all_sensorloc,dlens,p,j)
                    i_piv = np.argmax(dlens_updated)
                    dlen = dlens_updated[i_piv]
                    dlens_old = dlens_updated
            else:
                i_piv = np.argmax(dlens)
                dlen = dlens[i_piv]

            # Choose pivot
            # i_piv = np.argmax(dlens_updated)

            # dlen = dlens_updated[i_piv]

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

if __name__ == '__main__':
    faces = datasets.fetch_olivetti_faces(shuffle=True)
    X = faces.data

    # n_samples, n_features = X.shape
    X_small = X[:,:256]
    n_samples, n_features = X_small.shape
    print('Number of samples:', n_samples)
    print('Number of features (sensors):', n_features)

    # Global centering
    X_small = X_small - X_small.mean(axis=0)

    # Local centering
    X_small -= X_small.mean(axis=1).reshape(n_samples, -1)

    n_row, n_col = 2, 3
    n_components = n_row * n_col
    image_shape = (16, 16)
    nx = 16
    ny = 16

    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
        '''Function for plotting faces'''
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                    interpolation='nearest',
                    vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

   # plot_gallery("First few centered faces", X[:n_components])

    #Find all sensor locations using built in QR optimizer
    #max_const_sensors = 230
    n_const_sensors = 0
    n_sensors = 10
    n_modes = 10
    r = 5
    # dmd = DMD(svd_rank=0,exact=True,opt=False)
    # dmd.fit(X.T)
    # U = dmd.modes.real
    # np.shape(U)
    # max_basis_modes = 200

    # model_dmd_unconstrained = ps.SSPOR(n_sensors=n_sensors, basis=ps.basis.Custom(n_basis_modes=n_modes, U=U))
    # model_dmd_unconstrained.fit(X)
    # basis_matrix_dmd = model_dmd_unconstrained.basis_matrix_

    # all_sensors_dmd_unconstrained = model_dmd_unconstrained.get_all_sensors()
    # top_sensors_dmd_unconstrained = model_dmd_unconstrained.get_selected_sensors()
    # optimality_dmd = ps.utils._validation.determinant(top_sensors_dmd_unconstrained, n_features, basis_matrix_dmd)
    # print(optimality0)
    basis = ps.basis.SVD(n_basis_modes=n_modes)
    optimizer  = ps.optimizers.QR()
    model = ps.SSPOR(optimizer=optimizer, n_sensors=n_sensors, basis=basis)
    model.fit(X_small)
    top_sensors0 = model.get_selected_sensors()
    all_sensors = model.get_all_sensors()
    # basis_matrix0 = model.basis_matrix_
    # optimality0 = ps.utils._validation.determinant(top_sensors0, n_features, basis_matrix0)
    # print(optimality0)
    ##Constrained sensor location on the grid:
    xmin = 0
    xmax = 64
    ymin = 10
    ymax = 30
    sensors_constrained = ps.utils._constraints.get_constraind_sensors_indices(xmin,xmax,ymin,ymax,nx,ny,all_sensors) #Constrained column indices
    # didx = np.isin(all_sensors,sensors_constrained,invert=False)
    # const_index = np.nonzero(didx)
    # j =



    #Plotting the constrained region
    ax = plt.subplot()
    #Plot constrained space
    img = np.zeros(n_features)
    img[sensors_constrained] = 1
    im = plt.imshow(img.reshape(image_shape),cmap=plt.cm.binary)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title('Constrained region')

    ## Fit the dataset with the optimizer GQR
    optimizer1 = GQR(sensors_constrained,n_sensors,n_const_sensors,all_sensors, constraint_option = "radii_constraints",nx = nx, ny = ny, r = r)
    model1 = ps.SSPOR( optimizer = optimizer1, n_sensors = n_sensors, basis=basis)
    model1.fit(X_small)
    all_sensors1 = model1.get_all_sensors()
    basis_matrix = model1.basis_matrix_
    top_sensors = model1.get_selected_sensors()
    print(top_sensors)
    optimality = ps.utils._validation.determinant(top_sensors, n_features, basis_matrix)
    print(optimality)
    # optimizer_dmd_constrained = ps.optimizers.GQR(sensors_constrained,n_sensors,n_const_sensors,all_sensors_dmd_unconstrained,constraint_option = "exact_n_const_sensors",nx = nx, ny = ny, r = r)
    # model_dmd_constrained = ps.SSPOR(n_sensors=n_sensors, basis=ps.basis.Custom(n_basis_modes=n_modes, U=U), optimizer = optimizer_dmd_constrained)
    # model_dmd_constrained.fit(X)
    # all_sensors_dmd_constrained = model_dmd_constrained.get_all_sensors()

    # top_sensors_dmd_constrained = model_dmd_constrained.get_selected_sensors()
    # basis_matrix_dmd_constrained = model_dmd_constrained.basis_matrix_
    # optimality = ps.utils._validation.determinant(top_sensors_dmd_constrained, n_features, basis_matrix_dmd_constrained)
    # print(optimality)

    ## TODO: this can be done using ravel and unravel more elegantly
    #yConstrained = np.floor(top_sensors[:n_const_sensors]/np.sqrt(n_features))
    #xConstrained = np.mod(top_sensors[:n_const_sensors],np.sqrt(n_features))

    # img = np.zeros(n_features)
    # img[top_sensors_dmd_constrained] = 16
    # #plt.plot(xConstrained,yConstrained,'*r')
    # plt.plot([xmin,xmin],[ymin,ymax],'r')
    # plt.plot([xmin,xmax],[ymax,ymax],'r')
    # plt.plot([xmax,xmax],[ymin,ymax],'r')
    # plt.plot([xmin,xmax],[ymin,ymin],'r')
    # plt.imshow(img.reshape(image_shape),cmap=plt.cm.binary)
    # plt.title('n_sensors = {}, n_constr_sensors = {}'.format(n_sensors,n_const_sensors))
    # plt.show()

    img = np.zeros(n_features)
    img[top_sensors] = 16
    fig,ax = plt.subplots(1)

    ax.imshow(img.reshape(image_shape),cmap=plt.cm.binary)
    print(top_sensors)
    top_sensors_grid = np.unravel_index(top_sensors, (nx,ny))
    # figure, axes = plt.subplots()
    for i in range(len(top_sensors_grid[0])):
        circ = Circle( (top_sensors_grid[1][i], top_sensors_grid[0][i]), r ,color='r',fill = False )
        ax.add_patch(circ)
    # ax.plot([xmin,xmin],[ymin,ymax],'-r')
    # ax.plot([xmax,xmax],[ymin,ymax],'-r')
    # ax.plot([xmin,xmax],[ymin,ymin],'-r')
    # ax.plot([xmin,xmax],[ymax,ymax],'-r')
    ax.set_aspect('equal')
    # ax.set_xlim([0,64])
    # ax.set_ylim([0,64])
    plt.show()


