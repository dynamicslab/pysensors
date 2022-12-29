
"""
Various utility functions for mapping constrained sensors locations with the column indices for class GQR.
"""

import numpy as np
import sys, os


def get_constraind_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix.

    Parameters
    ----------
    x_min: int, lower bound for the x-axis constraint
    x_max : int, upper bound for the x-axis constraint
    y_min : int, lower bound for the y-axis constraint
    y_max : int, upper bound for the y-axis constraint
    nx : int, image pixel (x dimensions of the grid)
    ny : int, image pixel (y dimensions of the grid)
    all_sensors : np.ndarray, shape [n_features], ranked list of sensor locations.

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix.
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

def get_constrained_sensors_indices_linear(x_min, x_max, y_min, y_max,df):
    """
    Function for obtaining constrained column indices from already existing linear sensor locations on the grid.

    Parameters
    ----------
    x_min: int, lower bound for the x-axis constraint
    x_max : int, upper bound for the x-axis constraint
    y_min : int, lower bound for the y-axis constraint
    y_max : int, upper bound for the y-axis constraint
    df : pandas.DataFrame, a dataframe containing the features  and samples

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix.
    """
    x = df['X (m)'].to_numpy()
    n_features = x.shape[0]
    y = df['Y (m)'].to_numpy()
    idx_constrained = []
    for i in range(n_features):
        if (x[i] >= x_min and x[i] <= x_max) and (y[i] >= y_min and y[i] <= y_max):
            idx_constrained.append(i)
    return idx_constrained

def functional_constraints(functionHandler, idx,kwargs):
    """
    Function for evaluating the functional constraints.

    Parameters
    ----------
    functionHandler : function, a function handle to the function which is to be evaluated
    idx :

    Return
    ------

    """
    shape = kwargs['shape']
    xLoc,yLoc = get_coordinates_from_indices(idx,shape)
    functionName = os.path.basename(functionHandler).strip('.py')
    dirName = os.path.dirname(functionHandler)
    sys.path.insert(0,os.path.expanduser(dirName))
    module = __import__(functionName)
    func = getattr(module, functionName)
    g = func(xLoc, yLoc,**kwargs)
    return g

def constraints_eval(constraints,senID,**kwargs):
    """_summary_

    Args:
        constraints (_type_): _description_
    """
    nConstraints = len(constraints)
    G = np.zeros((len(senID),nConstraints))
    for i in range(nConstraints):
        G[:,i] = functional_constraints(constraints[i],senID,kwargs)
    return G

def get_functionalConstraind_sensors_indices(senID,g):
    """
    Function for mapping constrained sensor locations on the grid with the column indices of the basis_matrix.

    Parameters
    ----------
    senID: int, sensor ID
    g : float, constraint evaluation function (negative if violating the constraint)

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix.
    """

    idx_constrained = senID[g<0].tolist()
    rank = np.where(np.isin(senID,idx_constrained)==True)[0].tolist()
    return idx_constrained, rank

def order_constrained_sensors(idx_constrained_list, ranks_list):
    sortedConstraints,ranks =zip(*[[x,y] for x,y in sorted(zip(idx_constrained_list, ranks_list),key=lambda x: (x[1]))])
    return sortedConstraints,ranks

def get_coordinates_from_indices(idx,shape):
    return np.unravel_index(idx,shape,'F')

def get_indices_from_coordinates(corrdinates,shape):
    return np.ravel_multi_index(corrdinates,shape,order='F')

if __name__ == '__main__':

    import pysensors as ps
    from sklearn import datasets

    # Test the constraintsEval function
    const1 = '~/projects/pysensors/examples/userExplicitConstraint1.py'
    const2 = '~/projects/pysensors/examples/userExplicitConstraint2.py'
    constList = [const1, const2]
    faces = datasets.fetch_olivetti_faces(shuffle=True)
    XX = faces.data
    n_samples, n_features = XX.shape
    # Global centering
    XX = XX - XX.mean(axis=0)
    # Local centering
    XX -= XX.mean(axis=1).reshape(n_samples, -1)

    n_sensors0 = 15
    n_modes0 = 15
    basis1 = ps.basis.SVD(n_basis_modes=n_modes0)
    optimizer_faces = ps.optimizers.QR()
    model = ps.SSPOR(basis=basis1,optimizer=optimizer_faces, n_sensors=n_sensors0)
    model.fit(XX)
    basis_matrix = model.basis_matrix_

    all_sensors0 = model.get_all_sensors()
    top_sensors0 = model.get_selected_sensors()

    xTopUnc = np.mod(top_sensors0,np.sqrt(n_features))
    yTopUnc = np.floor(top_sensors0/np.sqrt(n_features))
    xAllUnc = np.mod(all_sensors0,np.sqrt(n_features))
    yAllUnc = np.floor(all_sensors0/np.sqrt(n_features))

    # sensors_constrained = ps.utils._constraints.get_constraind_sensors_indices(xmin,xmax,ymin,ymax,nx,ny,all_sensors0) #Constrained column indices
    G = ps.utils._constraints.constraints_eval(constList,top_sensors0,shape=(64,64))
    idx_constrainedConst,ranks = ps.utils._constraints.get_functionalConstraind_sensors_indices(top_sensors0,G[:,0])
    idx_constrainedConst2,rank2 = ps.utils._constraints.get_functionalConstraind_sensors_indices(top_sensors0,G[:,1])

    idx_constrainedConst.extend(idx_constrainedConst2)
    ranks.extend(rank2)
    idx_constr_sorted, ranks = ps.utils._constraints.order_constrained_sensors(idx_constrainedConst,ranks)

    n_const_sensors0 = 1
    optimizer1 = ps.optimizers.GQR()
    opt_kws={'idx_constrained':idx_constrainedConst,
             'n_sensors':n_sensors0,
             'n_const_sensors':n_const_sensors0,
             'all_sensors':all_sensors0,
             'constraint_option':"max_n"}
    model1 = ps.SSPOR(basis = basis1, optimizer = optimizer1, n_sensors = n_sensors0)
    model1.fit(XX,**opt_kws)
    basis_matrix_svd = model1.basis_matrix_
    all_sensors1 = model1.get_all_sensors()

    top_sensors = model1.get_selected_sensors()
    print(top_sensors)
    dterminant_faces_svd = ps.utils._validation.determinant(top_sensors,n_features,basis_matrix_svd)
    print(dterminant_faces_svd)


    const3 = '/Users/abdomg/projects/Sparse_Sensing_in_NDTs_LDRD/notebooks/myBoxConstraint.py'
    constList2 =[const3]
    constr_kws = {'xmin':10,'xmax':30,'ymin':20,'ymax':40,'shape':(64,64)}
    G2 = ps.utils._constraints.constraints_eval(constList2,all_sensors0,**constr_kws)