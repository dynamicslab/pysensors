
"""
Various utility functions for mapping constrained sensors locations with the column indices for class GQR.
"""

import numpy as np


def get_constraind_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix.

    Parameters
        ----------
        x_min: int,
            Lower bound for the x-axis constraint
        x_max : int,
            Upper bound for the x-axis constraint
        y_min : int,
            Lower bound for the y-axis constraint
        y_max : int
            Upper bound for the y-axis constraint
        nx : int
            Image pixel (x dimensions of the grid)
        ny : int
            Image pixel (y dimensions of the grid)
        all_sensors : np.ndarray, shape [n_features]
            Ranked list of sensor locations.

        Returns
        -------
        idx_constrained : np.darray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
    """
    n_features = len(all_sensors)
    image_size = int(np.sqrt(n_features))
    a = np.unravel_index(all_sensors, (nx,ny))
    constrained_sensorsx = []
    constrained_sensorsy = []
    for i in range(n_features):
        if (a[0][i] >= x_min and a[0][i] <= x_max) and (a[1][i] >= y_min and a[1][i] <= y_max):  
            constrained_sensorsx.append(a[0][i])
            constrained_sensorsy.append(a[1][i])

    constrained_sensorsx = np.array(constrained_sensorsx)
    constrained_sensorsy = np.array(constrained_sensorsy)
    constrained_sensors_array = np.stack((constrained_sensorsy, constrained_sensorsx), axis=1)
    constrained_sensors_tuple = np.transpose(constrained_sensors_array)
    if len(constrained_sensorsx) == 0: ##Check to handle condition when number of sensors in the constrained region = 0
        idx_constrained = []
    else:
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (nx,ny))
    return idx_constrained

def get_constrained_sensors_indices_linear(x_min,x_max,y_min,y_max,df):
    """
    Function for obtaining constrained column indices from already existing linear sensor locations on the grid.

    Parameters
        ----------
        x_min: int,
            Lower bound for the x-axis constraint
        x_max : int,
            Upper bound for the x-axis constraint
        y_min : int,
            Lower bound for the y-axis constraint
        y_max : int
            Upper bound for the y-axis constraint
        df : pandas.DataFrame
            A dataframe containing the features  and samples
        
        Returns
        -------
        idx_constrained : np.darray, shape [No. of constrained locations]
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
    """
    x = df['X (m)'].to_numpy()
    n_features = x.shape[0]
    y = df['Y (m)'].to_numpy()
    idx_constrained = []
    for i in range(n_features):
        if (x[i] >= x_min and x[i] <= x_max) and (y[i] >= y_min and y[i] <= y_max):
            idx_constrained.append(i)
    return idx_constrained

def box_constraints(position,lower_bound,upper_bound,):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix. ##TODO : BETTER DEFINITION

    Parameters
        ----------
        position: ##TODO: FILL
            
        lower_bound : ##TODO: FILL
           
        upper_bound : ##TODO: FILL
        
        Returns
        -------
        idx_constrained : np.darray, shape [No. of constrained locations]       ##TODO: CHECK IF CORRECT
            Array which contains the constrained locationsof the grid in terms of column indices of basis_matrix.
    """
    for i,xi in enumerate(position):
        f1 = position[i] - lower_bound[i]
        f2 = upper_bound[i] - position [i]
    return +1 if (f1 and f2 > 0) else -1

def functional_constraints(position, func_response,func_input, free_term):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix. ##TODO: BETTER DEFINITION

    Parameters
        ----------
        position: ##TODO : FILL
            
        func_response : ##TODO : FILL
            
        func_input: ##TODO : FILL
            
        free_term : ##TODO : FILL
        
        Returns
        -------
        g : ##TODO : FILL
            
    """
    g = func_response + func_input + free_term
    return g