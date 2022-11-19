"""
Various utility functions for calculating the norm and providing dlens_updated based on the different types of adaptive constraints for _gqr.py in optimizers.
"""

import numpy as np
from ..utils._constraints import get_constraind_sensors_indices_radii

def exact_n(lin_idx, dlens, piv, j, n_const_sensors, **kwargs): ##Will first force sensors into constrained region
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

def max_n(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
    """
    Function for mapping constrained sensor locations with the QR procedure (Optimally).

    Parameters
    ----------
    lin_idx: np.ndarray, shape [No. of constrained locations]
        Array which contains the constrained locations of the grid in terms of column indices of basis_matrix.
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

def predetermined(lin_idx, dlens, piv, j, n_const_sensors, **kwargs):
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
    ----------
    lin_idx: np.ndarray, shape [No. of constrained locations], array which contains
        the constrained locationsof the grid in terms of column indices of basis_matrix.
    dlens: np.ndarray, shape [Variable based on j], array which contains the norm of columns of basis matrix.
    piv: np.ndarray, shape [n_features], ranked list of sensor locations.
    n_const_sensors: int, number of sensors to be placed in the constrained area.
    j: int, iterative variable in the QR algorithm.

    Returns
    -------
    dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if 'n_sensors' in kwargs.keys():
        n_sensors = kwargs['n_sensors']
    else:
        raise ValueError("total number of sensors is not given!")

    didx = np.isin(piv[j:],lin_idx,invert=(n_sensors - n_const_sensors) <= j <= n_sensors)
    dlens[didx] = 0
    return dlens

__norm_calc_type = {}
__norm_calc_type['exact_n_const_sensors'] = exact_n
__norm_calc_type['max_n_const_sensors'] = max_n
__norm_calc_type['predetermined_norm_calc'] = predetermined

def returnInstance(cls, name):
  """
    Method designed to return class instance:
    Parameters
    ----------
    cls, class type
    name, string, name of class

    Returns
    -------
    __norm_calc_type[name], instance of class
  """
  if name not in __norm_calc_type:
    cls.raiseAnError (IOError, "{} NOT IMPLEMENTED!!!!!".format(name))
  return __norm_calc_type[name]