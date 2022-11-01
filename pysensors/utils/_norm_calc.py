"""
Various utility functions for calculating the norm and providing dlens_updated based on the different types of adaptive constraints for _gqr.py in optimizers.
"""

import numpy as np

def norm_calc_exact_n_const_sensors(lin_idx, dlens, piv, j, n_const_sensors, **kwargs): ##Will first force sensors into constrained region
    # num_sensors should be fixed for each custom constraint (for now)
    # num_sensors must be <= size of constraint region
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
    didx = np.isin(piv[j:],lin_idx,invert=j<n_const_sensors)
    dlens[didx] = 0
    return dlens

def norm_calc_max_n_const_sensors(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
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
    if 'all_sensors' in kwargs.keys():
        all_sensors = kwargs['all_sensors']
    else:
        all_sensors = []
    if 'n_sensors' in kwargs.keys():
        n_sensors = kwargs['n_sensors']
    else:
        n_sensors = len(all_sensors)
    counter = 0
    mask = np.isin(all_sensors,lin_idx,invert=False)
    const_idx = all_sensors[mask]
    updated_lin_idx = const_idx[n_const_sensors:]
    for i in range(n_sensors):
        if np.isin(all_sensors[i],lin_idx,invert=False):
            counter += 1
            if counter < n_const_sensors:
                dlens = dlens
            else:
                didx = np.isin(piv[j:],updated_lin_idx,invert=False)
                dlens[didx] = 0
    return dlens

def predetermined_norm_calc(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
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
    if 'n_sensors' in kwargs.keys():
        n_sensors = kwargs['n_sensors']
    else:
        raise ValueError ('total number of sensors is not given!')

    didx = np.isin(piv[j:],lin_idx,invert=(n_sensors - n_const_sensors) <= j <= n_sensors)
    dlens[didx] = 0
    return dlens

def f_radii_constraint(j,dlens,dlens_old,piv,nx,ny,r, all_sensors, n_sensors):
    if j == 1:
        idx_constrained = get_constraind_sensors_indices_radii(j,piv,r, nx,ny, all_sensors)
        print(idx_constrained)
        didx = np.isin(piv[j:],idx_constrained,invert= False)
        dlens[didx] = 0
        return dlens
    else:
        result = np.where(dlens_old == 0)[0]
        result_list = result.tolist()
        result_list = [x + (j-1) for x in result_list]
        result_array = np.array(result_list)
        print(result_array)

        idx_constrained1 = get_constraind_sensors_indices_radii(j,piv,r, nx,ny, all_sensors)
        t = np.concatenate((idx_constrained1,result_array), axis = 0)
        didx = np.isin(piv[j:],t,invert= False)
        dlens[didx] = 0
        return dlens

def get_constraind_sensors_indices_radii(j,piv,r, nx,ny, all_sensors):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix.

    Parameters
        ----------
        all_sensors : np.ndarray, shape [n_features]
            Ranked list of sensor locations.

        Returns
        -------
        idx_constrained : np.darray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
    """
    n_features = len(all_sensors)
    image_size = int(np.sqrt(n_features))
    a = np.unravel_index(piv, (nx,ny))
    t = np.unravel_index(all_sensors, (nx,ny))
    x_cord = a[0][j-1]
    y_cord = a[1][j-1]
    #print(x_cord,y_cord)
    constrained_sensorsx = []
    constrained_sensorsy = []
    for i in range(n_features):
        if ((t[0][i]-x_cord)**2 + (t[1][i]-y_cord)**2) < r**2:
            constrained_sensorsx.append(t[0][i])
            constrained_sensorsy.append(t[1][i])

    constrained_sensorsx = np.array(constrained_sensorsx)
    constrained_sensorsy = np.array(constrained_sensorsy)
    constrained_sensors_array = np.stack((constrained_sensorsx, constrained_sensorsy), axis=1)
    constrained_sensors_tuple = np.transpose(constrained_sensors_array)
    if len(constrained_sensorsx) == 0: ##Check to handle condition when number of sensors in the constrained region = 0
        idx_constrained = []
    else:
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (nx,ny))
    return idx_constrained


__norm_calc_type = {}
__norm_calc_type['exact_n_const_sensors'] = norm_calc_exact_n_const_sensors
__norm_calc_type['max_n_const_sensors'] = norm_calc_max_n_const_sensors
__norm_calc_type['predetermined_norm_calc'] = predetermined_norm_calc

def returnInstance(cls, name):
  """
    Method designed to return class instance:
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __crossovers[name], instance of class
  """
  if name not in __norm_calc_type:
    cls.raiseAnError (IOError, "{} NOT IMPLEMENTED!!!!!".format(name))
  return __norm_calc_type[name]