"""
Various utility functions for calculating the norm and providing dlens_updated based on
the different types of adaptive constraints for _gqr.py in optimizers.
"""

import numpy as np
import pandas as pd

from pysensors.utils._constraints import (
    get_constrained_sensors_indices_distance,
    get_constrained_sensors_indices_distance_df,
)


def unconstrained(dlens, piv, j, **kwargs):
    return dlens


def exact_n(dlens, piv, j, **kwargs):
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
    ----------
    lin_idx: np.ndarray, shape [No. of constrained locations]
        Array which contains the constrained locations of the grid in terms of
        column indices of basis_matrix.
    dlens: np.ndarray, shape [n_features - j]
        Array which contains the norm of columns of basis matrix.
    piv: np.ndarray, shape [n_features]
        Ranked list of sensor locations.
    n_const_sensors: int,
        Number of sensors to be placed in the constrained area.
    j: int,
        current sensor to be placed in the QR/GQR algorithm.

    Returns
    -------
    dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if "idx_constrained" in kwargs.keys():
        lin_idx = kwargs["idx_constrained"]
    else:
        lin_idx = []
    if "n_const_sensors" in kwargs.keys():
        n_const_sensors = kwargs["n_const_sensors"]
    else:
        n_const_sensors = []
    if "all_sensors" in kwargs.keys():
        all_sensors = kwargs["all_sensors"]
    else:
        all_sensors = []
    if "n_sensors" in kwargs.keys() and kwargs["n_sensors"] not in [None, 0]:
        n_sensors = kwargs["n_sensors"]
    else:
        n_sensors = len(all_sensors)
    count = np.count_nonzero(np.isin(all_sensors[:j], lin_idx, invert=False))
    if np.isin(all_sensors[:n_sensors], lin_idx, invert=False).sum() < n_const_sensors:
        if n_sensors > j >= (n_sensors - (n_const_sensors - count)):
            didx = np.isin(piv[j:], lin_idx, invert=True)
            dlens[didx] = 0
    else:
        dlens = max_n(dlens, piv, j, **kwargs)
    return dlens


def max_n(dlens, piv, j, **kwargs):
    """
    Function for mapping constrained sensor locations with the QR procedure (Optimally).

    Parameters
    ----------
    lin_idx: np.ndarray, shape [No. of constrained locations]
        Array which contains the constrained locations of the grid in terms of column
        indices of basis_matrix.
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
    if "idx_constrained" in kwargs.keys():
        lin_idx = kwargs["idx_constrained"]
    else:
        lin_idx = []
    if "n_const_sensors" in kwargs.keys():
        n_const_sensors = kwargs["n_const_sensors"]
    else:
        n_const_sensors = []
    if "all_sensors" in kwargs.keys():
        all_sensors = kwargs["all_sensors"]
    else:
        all_sensors = []
    if "n_sensors" in kwargs.keys() and kwargs["n_sensors"] not in [None, 0]:
        n_sensors = kwargs["n_sensors"]
    else:
        n_sensors = len(all_sensors)
    counter = 0
    mask = np.isin(all_sensors, lin_idx, invert=False)
    const_idx = all_sensors[mask]
    updated_lin_idx = const_idx[n_const_sensors:]
    for i in range(n_sensors):
        if np.isin(all_sensors[i], lin_idx, invert=False):
            counter += 1
            if counter > n_const_sensors:
                didx = np.isin(piv[j:], updated_lin_idx, invert=False)
                dlens[didx] = 0
    return dlens


def predetermined(dlens, piv, j, **kwargs):
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
    ----------
    lin_idx: np.ndarray, shape [No. of constrained locations], array which contains
        the constrained locationsof the grid in terms of column indices of basis_matrix.
    dlens: np.ndarray, shape [Variable based on j], array which contains the norm of
    columns of basis matrix.
    piv: np.ndarray, shape [n_features], ranked list of sensor locations.
    n_const_sensors: int, number of sensors to be placed in the constrained area.
    j: int, iterative variable in the QR algorithm.

    Returns
    -------
    dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if "idx_constrained" in kwargs.keys():
        lin_idx = kwargs["idx_constrained"]
    else:
        lin_idx = []
    if "n_const_sensors" in kwargs.keys():
        n_const_sensors = kwargs["n_const_sensors"]
    else:
        n_const_sensors = []
    if "n_sensors" in kwargs.keys():
        n_sensors = kwargs["n_sensors"]
    else:
        raise ValueError("total number of sensors is not given!")

    didx = np.isin(
        piv[j:], lin_idx, invert=(n_sensors - n_const_sensors) <= j <= n_sensors
    )
    dlens[didx] = 0
    return dlens


def distance(dlens, piv, j, **kwargs):
    """
    Optimized distance constraint function.

    Parameters
    ----------
    dlens : np.ndarray
        Array of distance values
    piv : np.ndarray
        Array of sensor indices in order of placement
    j : int
        Current iteration (0-indexed)
    **kwargs : dict
        Additional parameters:
        - idx_constrained : list, optional
            List of constrained indices
        - n_const_sensors : int, optional
            Number of constrained sensors
        - all_sensors : np.ndarray, required
            Ranked list of sensor locations
        - info : np.ndarray or pd.DataFrame, required
            Data structure containing sensor information
        - r : float, required
            Radius constraint (minimum distance between sensors)
        - nx, ny : int, required if info is np.ndarray
            Grid dimensions
        - X_axis, Y_axis : str, required if info is pd.DataFrame
            Column names for X and Y coordinates in the DataFrame

    Returns
    -------
    dlens : np.ndarray
        Updated array with constrained locations marked as 0
    """
    idx_constrained = kwargs.get("idx_constrained", [])
    n_const_sensors = kwargs.get("n_const_sensors", [])  # noqa: F841
    all_sensors = kwargs.get("all_sensors", [])
    n_sensors = kwargs.get("n_sensors", len(all_sensors))  # noqa: F841

    if "info" not in kwargs:
        raise ValueError("Must provide 'info' parameter as a np.darray or dataframe")
    info = kwargs.get("info")

    if "r" not in kwargs:
        raise ValueError("Must provide 'r' parameter for radius constraints")
    r = kwargs.get("r")
    if r <= 0:
        raise ValueError(f"Radius 'r' must be positive, got {r}")
    if isinstance(info, np.ndarray):
        if "nx" not in kwargs:
            raise ValueError("Must provide nx parameter")
        nx = kwargs.get("nx")
        if "ny" not in kwargs:
            raise ValueError("Must provide nx parameter")
        ny = kwargs.get("ny")
        if j == 0:
            idx_constrained = get_constrained_sensors_indices_distance(
                j, piv, r, nx, ny, all_sensors
            )
            didx = np.isin(piv[j:], idx_constrained)
            dlens[didx] = 0
        else:
            constrained_mask = np.zeros(len(piv[j:]), dtype=bool)
            future_coords = np.unravel_index(piv[j:], (nx, ny))
            for i in range(j):
                sensor = piv[i]
                sensor_coords = np.unravel_index([sensor], (nx, ny))
                x_sensor, y_sensor = sensor_coords[0][0], sensor_coords[1][0]
                distances_sq = (future_coords[0] - x_sensor) ** 2 + (
                    future_coords[1] - y_sensor
                ) ** 2
                constrained_mask = constrained_mask | (distances_sq < r**2)
            dlens[constrained_mask] = 0

    elif isinstance(info, pd.DataFrame):
        if "X_axis" in kwargs.keys():
            X_axis = kwargs["X_axis"]
        else:
            raise Exception(
                "Must provide X_axis as **kwargs as your data is a dataframe"
            )
        if "Y_axis" in kwargs.keys():
            Y_axis = kwargs["Y_axis"]
        else:
            raise Exception(
                "Must provide Y_axis as **kwargs as your data is a dataframe"
            )

        if j == 0:
            idx_constrained = get_constrained_sensors_indices_distance_df(
                j, piv, r, info, all_sensors, X_axis, Y_axis
            )
            didx = np.isin(piv[j:], idx_constrained)
            dlens[didx] = 0
        else:
            constrained_mask = np.zeros(len(piv[j:]), dtype=bool)
            future_indices = piv[j:]
            future_coords_df = info.loc[future_indices]
            for i in range(j):
                sensor_idx = piv[i]
                sensor_x = info.loc[sensor_idx, X_axis]
                sensor_y = info.loc[sensor_idx, Y_axis]
                distances_sq = (future_coords_df[X_axis] - sensor_x) ** 2 + (
                    future_coords_df[Y_axis] - sensor_y
                ) ** 2
                constrained_mask = constrained_mask | (distances_sq.values < r**2)

            dlens[constrained_mask] = 0

    else:
        raise ValueError("'info' parameter must be either np.ndarray or pd.DataFrame")

    return dlens


__norm_calc_type = {}
__norm_calc_type[""] = unconstrained
__norm_calc_type["exact_n"] = exact_n
__norm_calc_type["max_n"] = max_n
__norm_calc_type["predetermined"] = predetermined
__norm_calc_type["distance"] = distance


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
        raise NotImplementedError("{} NOT IMPLEMENTED!!!!!\n".format(name))
    return __norm_calc_type[name]
