"""
Various utility functions for validation and computing reconstruction scores and errors.
"""

import numpy as np
from scipy.sparse import lil_matrix


def determinant(top_sensors, n_features, basis_matrix):
    """
    Function for calculating |C.T phi.T C phi|.

    Parameters
    ----------
    top_sensors: np.darray,
        Column indices of choosen sensor locations
    n_features : int,
        No. of features of dataset
    basis_matrix : np.darray,
        The basis matrix calculated by model.basis_matrix_
    Returns
    -------
    optimality : Float,
        The dterminant value obtained.
    """

    p = len(top_sensors)  # Number of sensors
    n, r = np.shape(basis_matrix)  # state dimension X Number of modes
    c = lil_matrix((p, n), dtype=np.int8)

    for i in range(p):
        c[i, top_sensors[i]] = 1
    phi = basis_matrix
    theta = c @ phi
    if p == r:
        M_gamma = theta
    elif p > r:
        M_gamma = theta.T @ theta
    else:  # TODO
        # raise an error that p cannot be less than r
        pass
    optimality = abs(np.linalg.det(M_gamma))
    return optimality


def relative_reconstruction_error(data, prediction):
    """
    Function for calculating relative error between actual data and the reconstruction

    Parameters
        ----------
        data: np.darray,
            The actual data from the dataset evaluated
        prediction : np.darray,
            The predicted values from model.predict(X[:,top_sensors])
        Returns
        -------
        error_val : Float,
            The relative error calculated.
    """
    error_val = (np.linalg.norm((data - prediction) / np.linalg.norm(data))) * 100
    return error_val
