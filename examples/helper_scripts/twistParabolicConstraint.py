import numpy as np


def twistParabolicConstraint(x, y, **kwargs):
    """
    Function for evaluating constrained sensor locations on the grid by returning a
    negative value (if index is constrained) and a positive value
    (if index is unconstrained).

    Parameters
    ----------
    x: float, x coordinate of all grid-points considered for sensor placement
    y : float, y coordinate of all grid-points considered for sensor placement

    **kwargs : h : float, x-coordinate of the vertex of the parabola we want to be
                    constrained;
               k : float, y-coordinate of the vertex of the parabola we want to be
                    constrained;
               a : float, The x-coordinate of the focus of the parabola.

    Returns
    -------
    g : np.darray, shape [No. of grid points],
        A boolean array for every single grid point based on whether the grid point lies
        in the constrained region or not.
    """
    # make sure the length of x is the same as y
    assert len(x) == len(y)
    if ("h" not in kwargs.keys()) or (kwargs["h"] is None):
        kwargs["h"] = 0.025
    if ("k" not in kwargs.keys()) or (kwargs["k"] is None):
        kwargs["k"] = 0
    if ("a" not in kwargs.keys()) or (kwargs["a"] is None):
        kwargs["a"] = 100
    # initialize the constraint evaluation function g
    g1 = np.zeros(len(x), dtype=float) - 1
    g2 = np.zeros(len(x), dtype=float) - 1
    g = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        # circle of center (h,k)=(0,0) and radius 2.5 cm
        g1[i] = (kwargs["a"] * (x[i] - kwargs["h"]) ** 2) - (y[i] - kwargs["k"])
        #
        # Second constraint:
        g2[i] = y[i] - 0.2
        if bool(g1[i] >= 0) == bool(g2[i] >= 0):
            g[i] = (bool(g1[i] >= 0) and bool(g2[i] >= 0)) - 1
    return g
