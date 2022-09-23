"""
Various utility functions for calculating the norm and providing dlens_updated based on the different types of adaptive constraints for _gqr.py in optimizers.
"""

import numpy as np

def norm_calc_exact_n_const_sensors(lin_idx, dlens, piv, j, n_const_sensors): ##Will first force sensors into constrained region
    #num_sensors should be fixed for each custom constraint (for now)
    #num_sensors must be <= size of constraint region
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
        ----------
        lin_idx: np.ndarray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
        dlens: np.ndarray, shape [Variable based on j]
            Array which contains the norm of columns of basis matrix.
        piv: np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        n_const_sensors: int,
            Number of sensors to be placed in the constrained area.
        j: int,
            Iterative variable in the QR algorithm.

        Returns
        -------
        dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if j < n_const_sensors: # force sensors into constraint region
        #idx = np.arange(dlens.shape[0])
        #dlens[np.delete(idx, lin_idx)] = 0

        didx = np.isin(piv[j:],lin_idx,invert=True)
        dlens[didx] = 0
    else:
        didx = np.isin(piv[j:],lin_idx,invert=False)
        dlens[didx] = 0
    return dlens

def norm_calc_max_n_const_sensors(lin_idx, dlens, piv, j, const_sensors,all_sensors,n_sensors): ##Optimal sensor placement with constraints (will place sensors in  the order of QR)
    """
    Function for mapping constrained sensor locations with the QR procedure (Optimally).

    Parameters
        ----------
        lin_idx: np.ndarray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
        dlens: np.ndarray, shape [Variable based on j]
            Array which contains the norm of columns of basis matrix.
        piv: np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        j: int,
            Iterative variable in the QR algorithm.
        const_sensors: int,
            Number of sensors to be placed in the constrained area.
        all_sensors: np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        n_sensors: integer,
            Total number of sensors

        Returns
        -------
        dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    counter = 0
    mask = np.isin(all_sensors,lin_idx,invert=False)
    const_idx = all_sensors[mask]
    updated_lin_idx = const_idx[const_sensors:]
    for i in range(n_sensors):
        if np.isin(all_sensors[i],lin_idx,invert=False):
            counter += 1
            if counter < const_sensors:
                dlens = dlens
            else:
                didx = np.isin(piv[j:],updated_lin_idx,invert=False)
                dlens[didx] = 0
    return dlens

def predetermined_norm_calc(lin_idx, dlens, piv, j, n_const_sensors, n_sensors):
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
        ----------
        lin_idx: np.ndarray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
        dlens: np.ndarray, shape [Variable based on j]
            Array which contains the norm of columns of basis matrix.
        piv: np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        n_const_sensors: int,
            Number of sensors to be placed in the constrained area.
        j: int,
            Iterative variable in the QR algorithm.

        Returns
        -------
        dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if (n_sensors - n_const_sensors) <= j <= n_sensors: # force sensors into constraint region
        #idx = np.arange(dlens.shape[0])
        #dlens[np.delete(idx, lin_idx)] = 0

        didx = np.isin(piv[j:],lin_idx,invert=True)
        dlens[didx] = 0
    else:
        didx = np.isin(piv[j:],lin_idx,invert=False)
        dlens[didx] = 0
    return dlens

def f_radii_constraint(j,dlens,dlens_old,piv,nx,ny,r):
    a = np.unravel_index(piv, (nx,ny))
    n_features = len(piv)
    if j == 1:
        x_cord = a[0][j-1]
        y_cord = a[1][j-1]
        #print(x_cord, y_cord)
        constrained_sensorsx = []
        constrained_sensorsy = []
        for i in range(n_features):
            if ((a[0][i]-x_cord)**2 + (a[1][i]-y_cord)**2) < r**2: 
                constrained_sensorsx.append(a[0][i])
                constrained_sensorsy.append(a[1][i])
        constrained_sensorsx = np.array(constrained_sensorsx)
        constrained_sensorsy = np.array(constrained_sensorsy)
        constrained_sensors_array = np.stack((constrained_sensorsy, constrained_sensorsx), axis=1)
        constrained_sensors_tuple = np.transpose(constrained_sensors_array)
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (nx,ny))
#         print(idx_constrained)
        didx = np.isin(piv[j:],idx_constrained,invert= False)
        dlens[didx] = 0
        return dlens
    else: 
        result = np.where(dlens_old == 0)[0]
        result_list = result.tolist()
        result_list = [x - 1 for x in result_list]
        result_array = np.array(result_list)
        x_cord = a[0][j-1]
        y_cord = a[1][j-1]
        #print(x_cord, y_cord)
        constrained_sensorsx = []
        constrained_sensorsy = []
        for i in range(n_features):
            if ((a[0][i]-x_cord)**2 + (a[1][i]-y_cord)**2) < r**2: 
                constrained_sensorsx.append(a[0][i])
                constrained_sensorsy.append(a[1][i])
        constrained_sensorsx = np.array(constrained_sensorsx)
        constrained_sensorsy = np.array(constrained_sensorsy)
        constrained_sensors_array = np.stack((constrained_sensorsy, constrained_sensorsx), axis=1)
        constrained_sensors_tuple = np.transpose(constrained_sensors_array)
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (nx,ny))
        t = np.concatenate((idx_constrained,result_array), axis = 0)
        didx = np.isin(piv[j:],t,invert= False)
        dlens[didx] = 0
        return dlens