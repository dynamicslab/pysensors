"""
Various utility functions for mapping constrained sensors locations with the column
indices for class GQR.
"""

import operator
import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_constrained_sensors_indices(x_min, x_max, y_min, y_max, nx, ny, all_sensors):
    """
    Function for mapping constrained sensor locations on the grid with the column
    indices of the basis_matrix.

    Parameters
    ----------
    x_min: float, lower bound for the x-axis constraint
    x_max : float, upper bound for the x-axis constraint
    y_min : float, lower bound for the y-axis constraint
    y_max : float, upper bound for the y-axis constraint
    nx : int, image pixel (x dimensions of the grid)
    ny : int, image pixel (y dimensions of the grid)
    all_sensors : np.ndarray of integers, shape [n_features], ranked list of sensor
    locations.

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which
    contains the constrained
        locations of the grid in terms of column indices of basis_matrix.
    """
    if len(all_sensors) == 0:
        raise ValueError("all_sensors must be provided")
    if not np.issubdtype(all_sensors.dtype, np.integer):
        raise ValueError("all_sensors must be integers")
    if x_min >= x_max:
        raise ValueError("x_min must be less than x_max")
    if y_min >= y_max:
        raise ValueError("y_min must be less than y_max")
    if not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError("nx and ny must be integers")
    n_features = len(all_sensors)
    a = np.unravel_index(all_sensors, (nx, ny))
    constrained_sensorsx = []
    constrained_sensorsy = []
    for i in range(n_features):
        if (a[0][i] >= x_min and a[0][i] <= x_max) and (
            a[1][i] >= y_min and a[1][i] <= y_max
        ):
            constrained_sensorsx.append(a[0][i])
            constrained_sensorsy.append(a[1][i])

    constrained_sensorsx = np.array(constrained_sensorsx)
    constrained_sensorsy = np.array(constrained_sensorsy)
    constrained_sensors_array = np.stack(
        (constrained_sensorsy, constrained_sensorsx), axis=1
    )
    constrained_sensors_tuple = np.transpose(constrained_sensors_array)
    if (
        len(constrained_sensorsx) == 0
    ):  # Check to handle condition when number of sensors in the constrained region = 0
        idx_constrained = []
    else:
        idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (nx, ny))
    return idx_constrained


def get_constrained_sensors_indices_dataframe(x_min, x_max, y_min, y_max, df, **kwargs):
    """
    Function for obtaining constrained column indices from already existing linear
    sensor locations on the grid.

    Parameters
    ----------
    x_min: float, lower bound for the x-axis constraint
    x_max : float, upper bound for the x-axis constraint
    y_min : float, lower bound for the y-axis constraint
    y_max : float, upper bound for the y-axis constraint
    df : pandas.DataFrame, a dataframe containing the features  and samples

    Keyword Arguments
    -----------------
    X_axis : string,
        Name of the column in dataframe to be plotted on the X axis.
    Y-axis : string,
        Name of the column in dataframe to be plotted on the Y axis.
    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which
    contains the constrained locations of the grid in terms of column indices of
    basis_matrix.
    """
    if "X_axis" in kwargs.keys():
        X_axis = kwargs["X_axis"]
    else:
        raise Exception("Must provide X_axis as **kwargs as your data is a dataframe")
    if "Y_axis" in kwargs.keys():
        Y_axis = kwargs["Y_axis"]
    else:
        raise Exception("Must provide Y_axis as **kwargs as your data is a dataframe")
    if df.isnull().values.any():
        df = df.dropna()
    x = df[X_axis].to_numpy()
    n_features = x.shape[0]
    y = df[Y_axis].to_numpy()

    idx_constrained = []
    for i in range(n_features):
        if (x[i] >= x_min and x[i] < x_max) and (y[i] >= y_min and y[i] < y_max):
            idx_constrained.append(i)
    return idx_constrained


def get_constrained_sensors_indices_distance(j, piv, r, nx, ny, all_sensors):
    """
    Efficiently finds sensors within radius r of a given sensor.

    Parameters
    ----------
    j : int
        Current iteration (0-indexed)
    piv : np.ndarray
        Array of sensor indices in order of placement
    r : float
        Radius constraint (minimum distance between sensors)
    nx, ny : int
        Grid dimensions
    all_sensors : np.ndarray
        Ranked list of sensor locations.

    Returns
    -------
    idx_constrained : np.ndarray
        Array of sensor indices within radius r
    """
    sensor_idx = max(0, j - 1)
    current_sensor = piv[sensor_idx]
    current_coords = np.unravel_index([current_sensor], (nx, ny))
    x_cord, y_cord = current_coords[0][0], current_coords[1][0]
    sensor_coords = np.unravel_index(all_sensors, (nx, ny))
    distances_sq = (sensor_coords[0] - x_cord) ** 2 + (sensor_coords[1] - y_cord) ** 2
    return all_sensors[distances_sq < r**2]


def get_constrained_sensors_indices_distance_df(
    j, piv, r, df, all_sensors, X_axis, Y_axis
):
    """
    Efficiently finds sensors within radius r of a given sensor for DataFrame input.

    Parameters
    ----------
    j : int
        Current iteration (0-indexed)
    piv : np.ndarray
        Array of sensor indices in order of placement
    r : float
        Radius constraint (minimum distance between sensors)
    df : pd.DataFrame
        DataFrame containing sensor coordinates
    all_sensors : np.ndarray
        Ranked list of sensor locations
    X_axis : str
        Column name for X coordinates in the DataFrame
    Y_axis : str
        Column name for Y coordinates in the DataFrame

    Returns
    -------
    idx_constrained : np.ndarray
        Array of sensor indices within radius r
    """
    sensor_idx = max(0, j - 1)
    current_sensor = piv[sensor_idx]
    current_x = df.loc[current_sensor, X_axis]
    current_y = df.loc[current_sensor, Y_axis]
    sensors_df = df.loc[all_sensors]
    distances_sq = (sensors_df[X_axis] - current_x) ** 2 + (
        sensors_df[Y_axis] - current_y
    ) ** 2
    return all_sensors[distances_sq.values < r**2]


def load_functional_constraints(functionHandler):
    """
    Parameters:
    ----------
    functionHandler : The python file name that contains the constraint to be evaluated
    as a string

    Return
    -------
    Convert the functionHandler file into a callable function
    """
    functionName = os.path.basename(functionHandler).strip(".py")
    dirName = os.path.dirname(functionHandler)
    sys.path.insert(0, os.path.expanduser(dirName))
    module = __import__(functionName)
    func = getattr(module, functionName)
    return func


def order_constrained_sensors(idx_constrained_list, ranks_list):
    """
    Function for ordering constrained sensor locations on the grid according to their
    ranks.

    Parameters
    ----------
    idx_constrained_list : np.darray shape [No. of constrained locations], Constrained
    sensor locations
    ranks_list : np.darray shape [No. of constrained locations], Ranks of each
    constrained sensor location

    Returns
    -------
    sortedConstraints : np.darray, shape [No. of constrained locations], array which
    contains the constrained locations of the grid in terms of column indices of
    basis_matrix sorted according to their rank.
    ranks : np.darray, shape [No. of constrained locations], array which contains
    the ranks of constrained sensors.
    """
    if len(ranks_list) == 0 or len(idx_constrained_list) == 0:
        sortedConstraints = []
        ranks = []
    else:
        sortedConstraints, ranks = zip(
            *[
                [x, y]
                for x, y in sorted(
                    zip(idx_constrained_list, ranks_list), key=lambda x: (x[1])
                )
            ]
        )
    return sortedConstraints, ranks


def get_coordinates_from_indices(idx, info, **kwargs):
    """
    Function for obtaining the coordinates on a grid from column indices

    Parameters
    ----------
    idx :  int, sensor ID
    info : pandas.DataFrame/np.ndarray shape [n_features, n_samples], Dataframe or
    Matrix which represent the measurement data.

    Keyword Arguments
    -----------------
    X_axis : string,
        Name of the column in dataframe to be plotted on the X axis.
    Y-axis : string,
        Name of the column in dataframe to be plotted on the Y axis.
    Field : string,
        Name of the column in dataframe to be plotted as a contour map.

    Returns:
        (x,y) : tuple, The coordinates on the grid of each sensor.
    """
    if isinstance(info, np.ndarray):
        return np.unravel_index(
            idx, (int(np.sqrt(info.shape[1])), int(np.sqrt(info.shape[1]))), "F"
        )
    elif isinstance(info, pd.DataFrame):
        if set(idx).issubset(np.arange(0, len(info))) is False:
            raise Exception("Sensor ID must be within dataframe entries")
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
        if "Z_axis" in kwargs.keys() and kwargs["Z_axis"] is not None:
            Z_axis = kwargs["Z_axis"]
            z = info.loc[idx, Z_axis].values
        else:
            z = None
        x = info.loc[idx, X_axis].values
        y = info.loc[idx, Y_axis].values

        return (x, y, z) if z is not None else (x, y)


def get_indices_from_coordinates(coordinates, shape):
    """
    Function for obtaining the indices of columns/sensors from coordinates on a
    grid when data is in the form of a matrix

    Parameters
    ----------
    coordinates : tuple of array_like , (x,y) pair coordinates of sensor locations on
    the grid
    shape : tuple of ints, Shape of the matrix fed as data to the algorithm

    Returns
    -------
    np.ravel_multi_index(coordinates,shape,order='F') : np.ndarray, The indices of the
    sensors.
    """
    return np.ravel_multi_index(coordinates, shape, order="F")


class BaseConstraint(object):
    """
    A General class for handling various functional and user-defined constraint shapes.
    It extends the ability of constraint handling with various plotting and annotating
    functionalities while constraining various user-defined regions on the grid.

    @ authors: Niharika Karnik (@nkarnik2999), Mohammad Abdo (@Jimmy-INL)
    and Joshua Cogliati (@joshua-cogliati-inl)
    """

    def __init__(self, **kwargs):
        """
        Attributes
        ----------
        Keyword Arguments:
        ------------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        if "data" in kwargs.keys():
            self.data = kwargs["data"]
        else:
            raise Exception("Must provide data as **kwargs")
        if isinstance(self.data, pd.DataFrame):
            if "X_axis" in kwargs.keys():
                self.X_axis = kwargs["X_axis"]
            else:
                raise Exception(
                    "Must provide X_axis as **kwargs as your data is a dataframe"
                )
            if "Y_axis" in kwargs.keys():
                self.Y_axis = kwargs["Y_axis"]
            else:
                raise Exception(
                    "Must provide Y_axis as **kwargs as your data is a dataframe"
                )
            if "Z_axis" in kwargs.keys():
                self.Z_axis = kwargs["Z_axis"]
            else:
                self.Z_axis = None
            if "Field" in kwargs.keys():
                self.Field = kwargs["Field"]
            else:
                raise Exception(
                    "Must provide Field as **kwargs as your data is a dataframe"
                )

    def functional_constraints(func, idx, info, **kwargs):
        """
        Function for evaluating the functional constraints.

        Parameters
        ----------
        func : function, a function which is to be evaluated
        idx : np.ndarray, ranked list of sensor locations (column indices)
        info : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.

        Return
        ------
        g : function, Contains the function defined by the user for the functional
        constraint.
        """
        if isinstance(info, np.ndarray):
            xLoc, yLoc = get_coordinates_from_indices(idx, info)
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
            if "Field" in kwargs.keys():
                Field = kwargs["Field"]
            else:
                raise Exception(
                    "Must provide Field as **kwargs as your data is a dataframe"
                )
            if "Z_axis" in kwargs.keys():
                Z_axis = kwargs["Z_axis"]
            else:
                Z_axis = None
            xLoc, yLoc = get_coordinates_from_indices(
                idx, info, X_axis=X_axis, Y_axis=Y_axis, Z_axis=Z_axis, Field=Field
            )
        g = func(xLoc, yLoc, **kwargs)
        return g

    def get_functionalConstraind_sensors_indices(senID, g):
        """
        Function for finding constrained sensor locations on the grid and their ranks

        Parameters
        ----------
        senID: np.darray, ranked list of sensor locations (column indices)
        g : float, constraint evaluation function (negative if violating the constraint)

        Returns
        -------
        idx_constrained : np.darray, shape [No. of constrained locations], array which
        contains the constrained
            locations of the grid in terms of column indices of basis_matrix.
        rank : np.darray, shape [No. of constrained locations], array which contains
        rank of the constrained sensor locations
        """
        assert len(senID) == len(g)
        idx_constrained = senID[~g].tolist()
        rank = np.where(np.isin(idx_constrained, senID))[0].tolist()  # ==False
        return idx_constrained, rank

    def get_constraint_indices(self, all_sensors, info):
        """
        A function for computing indices which lie within the region constrained by
        the user
        Attributes
        ----------
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer
        info : pandas.DataFrame/np.ndarray shape [n_features, n_samples],
            Dataframe or Matrix which represent the measurement data.
        Returns
        -----------
        idx_const : np.darray, shape [No. of constrained locations],
            array which contains the constrained locations of the grid in terms of
            column indices of basis_matrix.
        rank : np.darray, shape [No. of constrained locations],
            array which contains rank of the constrained sensor locations
        """
        if isinstance(info, np.ndarray):
            coords = get_coordinates_from_indices(all_sensors, info)
        elif isinstance(info, pd.DataFrame):
            coords = get_coordinates_from_indices(
                all_sensors,
                info,
                X_axis=self.X_axis,
                Y_axis=self.Y_axis,
                Z_axis=self.Z_axis,
                Field=self.Field,
            )
        nDims, nPoints = np.shape(coords)
        g = np.zeros(nPoints, dtype=bool)
        for i in range(nPoints):
            g[i] = self.constraint_function(np.array(coords).reshape(nDims, -1)[:, i])
        idx_const, rank = BaseConstraint.get_functionalConstraind_sensors_indices(
            all_sensors, g
        )
        return idx_const, rank

    def draw_constraint(self, plot=None, **kwargs):
        """
        Function for drawing the constraint defined by the user
        """
        if plot is None:
            _, ax = plt.subplots()
        else:
            _, ax = plot
        self.draw(ax, **kwargs)

    def plot_constraint_on_data(self, plot_type, plot=None, **kwargs):
        """
        Function for plotting the user-defined constraint on the data
        Attributes
        ----------
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        plot_type : string,
            the type of plot used to display the data
            image : if the data is represented in the fprm of an image
            scatter: if the data can be represented with a scatter plot
            contour_map: if the data can be represented in the form of a contour map
        plot : to plot on an exisiting subplot, pass plot = (fig, ax),
                otherwise leave plot = None
        Returns
        -----------
        A plot of the constraint on top of the measurement data plot.
        """
        if plot is None:
            if isinstance(self, Cylinder):
                self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = plot
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 0.3
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = plt.cm.coolwarm
        if "s" not in kwargs.keys():
            kwargs["s"] = 1
        if "color" not in kwargs.keys():
            kwargs["color"] = "red"
        if plot_type == "image":
            image = self.data[1, :].reshape(1, -1)
            n_samples, n_features = self.data.shape
            image_shape = (int(np.sqrt(n_features)), int(np.sqrt(n_features)))
            for i, comp in enumerate(image):
                vmax = max(comp.max(), -comp.min())
                self.ax.imshow(
                    comp.reshape(image_shape),
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=-vmax,
                    vmax=vmax,
                )
        elif plot_type == "scatter":
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            self.ax.scatter(x_vals, y_vals, color=kwargs["color"], marker=".")
        elif plot_type == "scatter3D":
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            z_vals = self.data[self.Z_axis]
            self.ax.scatter(x_vals, y_vals, z_vals, color=kwargs["color"], marker=".")
        elif plot_type == "contour_map":
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            self.ax.scatter(
                x_vals,
                y_vals,
                c=self.data[self.Field],
                cmap=kwargs["cmap"],
                s=kwargs["s"],
                alpha=kwargs["alpha"],
            )
        elif plot_type == "contour_map3D":
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            z_vals = self.data[self.Z_axis]
            self.ax.scatter(
                x_vals,
                y_vals,
                z_vals,
                c=self.data[self.Field],
                cmap=kwargs["cmap"],
                s=kwargs["s"],
                alpha=kwargs["alpha"],
            )
        self.draw(self.ax, **kwargs)

    def plot_grid(self, all_sensors):
        """
        Function to plot the grid with data points that signify sensor locations
        to choose from
        Attributes
        ----------
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer

        Returns
        -----------
        A plot of the user defined grid showing all possible sensor locations
        """
        if isinstance(self.data, np.ndarray):
            n_samples, n_features = self.data.shape
            x_val, y_val = get_coordinates_from_indices(all_sensors, self.data)
            fig, ax = plt.subplots()
            ax.scatter(x_val, y_val, color="blue", marker=".")
        elif isinstance(self.data, pd.DataFrame):
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            fig, ax = plt.subplots()
            ax.scatter(x_vals, y_vals, color="blue", marker=".")

    def plot_selected_sensors(
        self, sensors, all_sensors, color_constrained="red", color_unconstrained="green"
    ):
        """
        Function to plot the sensor locations choosen during the optimization procedure.
        This function plots near-optimal sensors which are unconstrained sensor
        locations choosen by QR in the user defined color_unconstrained/green and
        sensors that are choosen through constraining certain regions of the grid
        in the under defined color_constrained/red.
        Attributes
        ----------
        sensors : np.darray,
            A ranked list of all sensor indices computed from QR/GQR/CCQR optimizer
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer
        color_constrained : string,
            The color the sensors that were selected due to the applied constraints
            should be plotted in
        color_unconstrained : string,
            The color the sensors that were a part of the near-optimal sensors choosen
            through unconstrained QR optimizer should be plotted in
        Returns
        -----------
        A plot of the user defined grid showing chosen sensor locations
        """
        n_samples, n_features = self.data.shape
        n_sensors = len(sensors)
        constrained = sensors[~np.isin(sensors, all_sensors[:n_sensors])]
        unconstrained = sensors[np.isin(sensors, all_sensors[:n_sensors])]

        if isinstance(self.data, np.ndarray):
            xconst = np.mod(constrained, np.sqrt(n_features))
            yconst = np.floor(constrained / np.sqrt(n_features))
            xunconst = np.mod(unconstrained, np.sqrt(n_features))
            yunconst = np.floor(unconstrained / np.sqrt(n_features))

            self.ax.plot(xconst, yconst, "*", color=color_constrained)
            self.ax.plot(xunconst, yunconst, "*", color=color_unconstrained)

        elif isinstance(self.data, pd.DataFrame):
            xconst, yconst = get_coordinates_from_indices(
                constrained,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )

            xunconst, yunconst = get_coordinates_from_indices(
                unconstrained,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )

            self.ax.plot(xconst, yconst, "*", color=color_constrained)
            self.ax.plot(xunconst, yunconst, "*", color=color_unconstrained)

    def sensors_dataframe(self, sensors):
        """
        Function to form a dataframe of the sensor index along with
        it's coordinate (X,Y,Z) positions
        Attributes
        ----------
        sensors : np.darray,
            A ranked list of all sensor indices choosen from QR/CCQR/GQR optimizer
        Returns
        -----------
        A dataframe of the sensor locations choosen
        """
        n_samples, n_features = self.data.shape
        n_sensors = len(sensors)
        if isinstance(self.data, np.ndarray):
            xTop = np.mod(sensors, np.sqrt(n_features))
            yTop = np.floor(sensors / np.sqrt(n_features))
        elif isinstance(self.data, pd.DataFrame):
            xTop, yTop = get_coordinates_from_indices(
                sensors,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )
        columns = ["Sensor ID", "SensorX", "sensorY"]
        Sensors_df = pd.DataFrame(
            data=np.vstack([sensors, xTop, yTop]).T, columns=columns, dtype=float
        )
        Sensors_df.head(n_sensors)
        return Sensors_df

    def annotate_sensors(
        self, sensors, all_sensors, color_constrained="red", color_unconstrained="green"
    ):
        """
        Function to annotate the sensor location on the grid while also plotting the
        sensor location

        Attributes
        ----------
        sensors : np.darray,
            A ranked list of all sensor indices choosen from QR/CCQR/GQR optimizer
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer
        color_constrained : string,
            The color the sensors that were selected due to the applied constraints
            should be plotted in
        color_unconstrained : string,
            The color the sensors that were a part of the near-optimal sensors choosen
            through unconstrained QR optimizer should be plotted in

        Returns
        -----------
        Annotation of sensor rank near the choosen sensor locations
        """
        n_samples, n_features = self.data.shape
        n_sensors = len(sensors)

        # Fixed logic for finding constrained and unconstrained sensors
        constrained = sensors[~np.isin(sensors, all_sensors[:n_sensors])]
        unconstrained = sensors[np.isin(sensors, all_sensors[:n_sensors])]

        if isinstance(self.data, np.ndarray):
            xTop = np.mod(sensors, np.sqrt(n_features))
            yTop = np.floor(sensors / np.sqrt(n_features))

            xconst = np.mod(constrained, np.sqrt(n_features))
            yconst = np.floor(constrained / np.sqrt(n_features))

            xunconst = np.mod(unconstrained, np.sqrt(n_features))
            yunconst = np.floor(unconstrained / np.sqrt(n_features))

            data = np.vstack([sensors, xTop, yTop]).T  # noqa:F841

            self.ax.plot(xconst, yconst, "*", color=color_constrained, alpha=0.5)
            self.ax.plot(xunconst, yunconst, "*", color=color_unconstrained, alpha=0.5)

            # Improved annotation logic with index checking
            for ind in range(len(sensors)):
                if ind < len(xTop) and ind < len(yTop):  # Make sure index is in bounds
                    self.ax.annotate(
                        f"{ind}",
                        (xTop[ind], yTop[ind]),
                        xycoords="data",
                        xytext=(-20, 20),
                        textcoords="offset points",
                        color="r",
                        fontsize=12,
                        arrowprops=dict(arrowstyle="->", color="black"),
                    )

        elif isinstance(self.data, pd.DataFrame):
            xTop, yTop = get_coordinates_from_indices(
                sensors,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )

            xconst, yconst = get_coordinates_from_indices(
                constrained,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )

            xunconst, yunconst = get_coordinates_from_indices(
                unconstrained,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )

            self.ax.plot(xconst, yconst, "*", color=color_constrained, alpha=0.5)
            self.ax.plot(xunconst, yunconst, "*", color=color_unconstrained, alpha=0.5)

            # Improved annotation logic - check array lengths and indices
            for i in range(len(sensors)):
                if i < len(xTop) and i < len(yTop):  # Make sure index is in bounds
                    # Check that the coordinates are valid (not NaN)
                    if np.isfinite(xTop[i]) and np.isfinite(yTop[i]):
                        self.ax.annotate(
                            f"{i}",
                            (xTop[i], yTop[i]),
                            xycoords="data",
                            xytext=(-20, 20),
                            textcoords="offset points",
                            color="r",
                            fontsize=12,
                            arrowprops=dict(arrowstyle="->", color="black"),
                        )


class Circle(BaseConstraint):
    """
    General class for dealing with circular user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(self, center_x, center_y, radius, loc="in", **kwargs):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        center_x : float,
            x-coordinate of the center of circle
        center_y : float,
            y-coordinate of the center of circle
        radius : float,
            radius of the circle
        loc : string- 'in'/'out',
            specifying whether the inside or outside of the shape is constrained

        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.loc = loc

    def draw(self, ax, **kwargs):
        """
        Function to plot a circle based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint circle should be plotted
        """
        if "fill" not in kwargs.keys():
            kwargs["fill"] = False
        if "color" not in kwargs.keys():
            kwargs["color"] = "r"
        if "lw" not in kwargs.keys():
            kwargs["lw"] = 2
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 1.0
        c = patches.Circle(
            (self.center_x, self.center_y),
            self.radius,
            fill=kwargs["fill"],
            color=kwargs["color"],
            lw=kwargs["lw"],
            alpha=kwargs["alpha"],
        )
        ax.add_patch(c)
        ax.autoscale_view()

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies
        inside/outside the defined constrained region
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether
            it lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether
            it lies inside or outside the constrained region
        """
        x, y = coords[:]
        inFlag = ((x - self.center_x) ** 2 + (y - self.center_y) ** 2) <= self.radius**2
        if self.loc.lower() == "in":
            return not inFlag
        else:
            return inFlag


class Cylinder(BaseConstraint):
    """
    General class for dealing with circular user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(
        self, center_x, center_y, center_z, radius, height, loc="in", **kwargs
    ):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        center_x : float,
            x-coordinate of the center of circle
        center_y : float,
            y-coordinate of the center of circle
        radius : float,
            radius of the circle
        loc : string- 'in'/'out',
            specifying whether the inside or outside of the shape is constrained

        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius
        self.height = height
        self.loc = loc
        if "axis" in kwargs.keys():
            self.axis = kwargs["axis"]
        else:
            self.axis = "Z_axis"

    def draw(self, ax, **kwargs):
        """
        Function to plot a cylinder based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint circle should be plotted
        """
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 0.3
            alpha = 3 * kwargs["alpha"]
        if kwargs["alpha"] * 3 < 0.5:
            alpha = 1.0
        else:
            alpha = kwargs["alpha"] * 3
        if "color" not in kwargs.keys():
            kwargs["color"] = "red"
        theta = np.linspace(0, 2 * np.pi, 100)
        if self.axis == "Z_axis":
            z = np.linspace(
                self.center_z - self.height / 2, self.center_z + self.height / 2, 100
            )
            theta, z = np.meshgrid(theta, z)
            x = self.center_x + self.radius * np.cos(theta)
            y = self.center_y + self.radius * np.sin(theta)
        elif self.axis == "X_axis":
            x = np.linspace(
                self.center_x - self.height / 2, self.center_x + self.height / 2, 100
            )
            theta, x = np.meshgrid(theta, x)
            y = self.center_y + self.radius * np.sin(theta)
            z = self.center_z + self.radius * np.cos(theta)
        else:
            y = np.linspace(
                self.center_y - self.height / 2, self.center_y + self.height / 2, 100
            )
            theta, y = np.meshgrid(theta, y)
            x = self.center_x + self.radius * np.cos(theta)
            z = self.center_z + self.radius * np.sin(theta)
        ax.plot_surface(x, y, z, alpha=alpha, color=kwargs["color"])
        ax.autoscale_view()

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies inside/outside
        the defined constrained region
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it
            lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it
            lies inside or outside the constrained region
        """
        x, y, z = coords[:]
        if isinstance(x, float):
            x, y, z = [x], [y], [z]
        nPoints = np.shape(np.array(coords).reshape(3, -1))[1]
        inFlag = np.zeros(nPoints, dtype=bool)
        for i in range(nPoints):
            if self.axis == "Z_axis":
                inFlag[i] = (
                    (
                        ((x[i] - self.center_x) ** 2 + (y[i] - self.center_y) ** 2)
                        <= self.radius**2
                    )
                    and self.center_z - self.height / 2 <= z[i]
                    and z[i] <= self.center_z + self.height / 2
                )
            elif self.axis == "Y_axis":
                inFlag[i] = (
                    (
                        ((x[i] - self.center_x) ** 2 + (z[i] - self.center_z) ** 2)
                        <= self.radius**2
                    )
                    and self.center_y - self.height / 2 <= y[i]
                    and y[i] <= self.center_y + self.height / 2
                )
            else:
                inFlag[i] = (
                    (
                        ((y[i] - self.center_y) ** 2 + (z[i] - self.center_z) ** 2)
                        <= self.radius**2
                    )
                    and self.center_x - self.height / 2 <= x[i]
                    and x[i] <= self.center_x + self.height / 2
                )
        if self.loc.lower() == "in":
            return map(operator.not_, inFlag)
        else:
            return inFlag


class Line(BaseConstraint):
    """
    General class for dealing with linear user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(self, x1, x2, y1, y2, **kwargs):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        x1 : float,
            x-coordinate of one end-point of the line
        x2 : float,
            x-coordinate of the other end-point of the line
        y1 : float,
            y-coordinate of one end-point of the line
        y2 : float,
            y-coordinate of the other end-point of the line

        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def draw(self, ax, **kwargs):
        """
        Function to plot a line based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint line should be plotted
        """
        if "color" not in kwargs.keys():
            kwargs["color"] = "r"
        if "lw" not in kwargs.keys():
            kwargs["lw"] = 2
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 1.0
        if "marker" not in kwargs.keys():
            kwargs["marker"] = None
        if "linestyle" not in kwargs.keys():
            kwargs["linestyle"] = "-"
        ax.plot(
            [self.x1, self.x2],
            [self.y1, self.y2],
            color=kwargs["color"],
            alpha=kwargs["alpha"],
            marker=kwargs["marker"],
            linestyle=kwargs["linestyle"],
        )

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies inside/outside
        the defined constrained region
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it
            lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it
            lies inside or outside the constrained region
        """
        x, y = coords[:]
        return (y - self.y1) * (self.x2 - self.x1) - (self.y2 - self.y1) * (
            x - self.x1
        ) >= 0


class Parabola(BaseConstraint):
    """
    General class for dealing with parabolic user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(self, h, k, a, loc, **kwargs):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        h : float,
            x-coordinate of the vertex of the parabola we want to be constrained
        k : float,
            y-coordinate of the vertex of the parabola we want to be constrained
        a : float,
            x-coordinate of the focus of the parabola
        loc : string- 'in'/'out',
            specifying whether the inside or outside of the shape is constrained

        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.h = h
        self.k = k
        self.a = a
        self.loc = loc

    def draw(self, ax, **kwargs):
        """
        Function to plot a parabola based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint parabola should be plotted
        """
        if isinstance(self.data, np.ndarray):
            grid_points = np.arange(self.data.shape[1])
            x, y = get_coordinates_from_indices(grid_points, self.data)
        elif isinstance(self.data, pd.DataFrame):
            grid_points = np.arange(len(self.data))
            x, y = get_coordinates_from_indices(
                grid_points,
                self.data,
                Y_axis=self.Y_axis,
                X_axis=self.X_axis,
                Field=self.Field,
            )
        y_vals = (self.a * ((x - self.h) ** 2)) - self.k
        ax.scatter(x, y_vals, s=1)

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies inside/outside
        the defined constrained region
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it lies
            inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it lies
            inside or outside the constrained region
        """
        x, y = coords[:]
        inFlag = (self.a * (x - self.h) ** 2) <= (y - self.k)
        if self.loc.lower() == "in":
            return not inFlag
        else:
            return inFlag


class Ellipse(BaseConstraint):
    """
    General class for dealing with elliptical user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(
        self, center_x, center_y, width, height, angle=0.0, loc="in", **kwargs
    ):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        center_x : float,
            x-coordinate of the center of circle
        center_y : float,
            y-coordinate of the center of circle
        width : float,
            total length (diameter) of horizontal axis.
        height : float,
            total length (diameter) of vertical axis.
        angle : float,
            angle of the orientation of the ellipse in degrees
        loc : string- 'in'/'out',
            specifying whether the inside or outside of the shape is constrained

        Keyword Arguments
        -----------------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.loc = loc
        self.angle = angle
        self.half_horizontal_axis = self.width / 2
        self.half_vertical_axis = self.height / 2

    def draw(self, ax, **kwargs):
        """
        Function to plot an ellipse based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint ellipse should be plotted
        """
        if "fill" not in kwargs.keys():
            kwargs["fill"] = False
        if "color" not in kwargs.keys():
            kwargs["color"] = "r"
        if "lw" not in kwargs.keys():
            kwargs["lw"] = 2
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 1.0
        c = patches.Ellipse(
            (self.center_x, self.center_y),
            self.width,
            self.height,
            angle=self.angle,
            fill=kwargs["fill"],
            color=kwargs["color"],
            lw=kwargs["lw"],
            alpha=kwargs["alpha"],
        )
        ax.add_patch(c)
        ax.autoscale_view()

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies inside/outside the
        defined constrained region
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it lies
            inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it lies
            inside or outside the constrained region
        """
        x, y = coords[:]
        angleInRadians = self.angle * np.pi / 180
        u = (x - self.center_x) * np.cos(angleInRadians) + (y - self.center_y) * np.sin(
            angleInRadians
        )
        v = -(x - self.center_x) * np.sin(angleInRadians) + (
            y - self.center_y
        ) * np.cos(angleInRadians)
        inFlag = (
            u**2 / self.half_horizontal_axis**2 + v**2 / self.half_vertical_axis**2 <= 1
        )
        if self.loc.lower() == "in":
            return not inFlag
        elif self.loc.lower() == "out":
            return inFlag


class Polygon(BaseConstraint):
    """
    General class for dealing with polygonal user defined constraints.
    Plotting, computing constraints functionalities included.
    """

    def __init__(self, xy_coords, loc="in", **kwargs):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        xy_coords : (N,2) array_like,
            an array consisting of tuples for (x,y) coordinates of points of the
            Polygon where N = No. of sides of the polygon
        """
        self.xy_coords = xy_coords
        self.loc = loc

    def draw(self, ax, **kwargs):
        """
        Function to plot a polygon based on user-defined coordinates
        Attributes
        ----------
        ax : axis on which the constraint polygon should be plotted
        """
        if "fill" not in kwargs.keys():
            kwargs["fill"] = False
        if "color" not in kwargs.keys():
            kwargs["color"] = "r"
        if "lw" not in kwargs.keys():
            kwargs["lw"] = 2
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 1.0
        c = patches.Polygon(
            self.xy_coords,
            fill=kwargs["fill"],
            color=kwargs["color"],
            lw=kwargs["lw"],
            alpha=kwargs["alpha"],
        )
        ax.add_patch(c)
        ax.autoscale_view()

    def constraint_function(self, coords):
        """
        Function to compute whether a certain point on the grid lies inside/outside
        the defined constrained region

        Attributes
        ----------
        coords : list or tuple
            [x, y] coordinates of point on the grid being evaluated to check whether
            it lies
            inside or outside the constrained region

        Returns
        -------
        bool
            True if point satisfies the constraint (inside for "in", outside for "out"),
            False otherwise
        """
        if len(coords) != 2:
            raise ValueError("coords must contain exactly 2 elements [x, y]")

        x, y = coords[:]
        polygon = self.xy_coords
        n = len(polygon)

        if n < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        inFlag = False

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                if y1 != y2:
                    x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    if x < x_intersect:
                        inFlag = not inFlag
        if self.loc.lower() == "in":
            return not inFlag
        elif self.loc.lower() == "out":
            return inFlag
        else:
            raise ValueError(f"Invalid constraint type: {self.loc}.Must be'in' or'out'")


class UserDefinedConstraints(BaseConstraint):
    """
    General class for dealing with any form of user defined constraints.
    The user can input the constraint in two forms:
    - As a python file which has the equation of the constraint the user wants to
    implement.
    - As a string with just the equation of the constraint the user wants to implement.
    Plotting, computing constraints functionalities included.
    """

    def __init__(self, all_sensors, **kwargs):
        super().__init__(**kwargs)
        """
        Attributes
        ----------
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer

        Keyword Arguments
        -----------------
        file : string,
            Name of the python file containing the equation of the constraint
        equation : string,
            Equation of the constraint the user wants to implement
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images)
            containing measurement data
        """
        self.all_sensors = all_sensors

        if "file" in kwargs.keys():
            self.file = kwargs["file"]
            self.functions = load_functional_constraints(self.file)
        else:
            self.file = None
        if "equation" in kwargs.keys():
            self.equations = [kwargs["equation"]]
        else:
            self.equations = None
        if self.equations is None and self.file is None:
            raise Exception("either file or equation should be provided")

        if isinstance(self.data, pd.DataFrame):
            if "X_axis" in kwargs.keys():
                self.X_axis = kwargs["X_axis"]
            else:
                raise Exception(
                    "Must provide X_axis as **kwargs as your data is a dataframe"
                )
            if "Y_axis" in kwargs.keys():
                self.Y_axis = kwargs["Y_axis"]
            else:
                raise Exception(
                    "Must provide Y_axis as **kwargs as your data is a dataframe"
                )
            if "Field" in kwargs.keys():
                self.Field = kwargs["Field"]
            else:
                raise Exception(
                    "Must provide either a python file ore equation of the constraint"
                )

    def draw(self, ax, **kwargs):
        """
        Function to plot the user-defined constraint
        Attributes
        ----------
        ax : axis on which the constraint should be plotted
        """
        if self.file is not None:
            nConstraints = len([self.functions])
            G = np.zeros((len(self.all_sensors), nConstraints), dtype=bool)
            for i in range(nConstraints):
                if isinstance(self.data, np.ndarray):
                    temp = BaseConstraint.functional_constraints(
                        self.functions, self.all_sensors, self.data
                    )
                    G[:, i] = [x > 0 for x in temp]
                    idx_const, rank = (
                        BaseConstraint.get_functionalConstraind_sensors_indices(
                            self.all_sensors, G[:, i]
                        )
                    )
                    x_val, y_val = get_coordinates_from_indices(idx_const, self.data)
                elif isinstance(self.data, pd.DataFrame):
                    temp = BaseConstraint.functional_constraints(
                        self.functions,
                        self.all_sensors,
                        self.data,
                        X_axis=self.X_axis,
                        Y_axis=self.Y_axis,
                        Field=self.Field,
                    )
                    G[:, i] = [x == 0 for x in temp]
                    idx_const, rank = (
                        BaseConstraint.get_functionalConstraind_sensors_indices(
                            self.all_sensors, G[:, i]
                        )
                    )
                    x_val, y_val = get_coordinates_from_indices(
                        idx_const,
                        self.data,
                        Y_axis=self.Y_axis,
                        X_axis=self.X_axis,
                        Field=self.Field,
                    )
        elif self.equations is not None:
            nConstraints = len(self.equations)
            G = np.zeros((len(self.all_sensors), nConstraints), dtype=bool)
            for i in range(nConstraints):
                if isinstance(self.data, np.ndarray):
                    xValue, yValue = get_coordinates_from_indices(
                        self.all_sensors, self.data
                    )
                    for k in range(len(xValue)):
                        G[k, i] = not eval(
                            self.equations[i], {"x": xValue[k], "y": yValue[k]}
                        )
                    idx_const, rank = (
                        BaseConstraint.get_functionalConstraind_sensors_indices(
                            self.all_sensors, G[:, i]
                        )
                    )
                    x_val, y_val = get_coordinates_from_indices(idx_const, self.data)
                elif isinstance(self.data, pd.DataFrame):
                    xValue, yValue = get_coordinates_from_indices(
                        self.all_sensors,
                        self.data,
                        Y_axis=self.Y_axis,
                        X_axis=self.X_axis,
                        Field=self.Field,
                    )
                    for k in range(len(xValue)):
                        G[k, i] = not eval(
                            self.equations[i], {"x": xValue[k], "y": yValue[k]}
                        )
                    idx_const, rank = (
                        BaseConstraint.get_functionalConstraind_sensors_indices(
                            self.all_sensors, G[:, i]
                        )
                    )
                    x_val, y_val = get_coordinates_from_indices(
                        idx_const,
                        self.data,
                        Y_axis=self.Y_axis,
                        X_axis=self.X_axis,
                        Field=self.Field,
                    )
        ax.scatter(x_val, y_val, s=1)

    def constraint(self):
        """
        Function to compute whether a certain point on the grid lies inside/outside the
        defined constrained region
        """
        if self.file is not None:
            nConstraints = len([self.functions])
            G = np.zeros((len(self.all_sensors), nConstraints), dtype=bool)
            for i in range(nConstraints):
                if isinstance(self.data, np.ndarray):
                    temp = BaseConstraint.functional_constraints(
                        self.functions, self.all_sensors, self.data
                    )
                    G[:, i] = [x >= 0 for x in temp]
                elif isinstance(self.data, pd.DataFrame):
                    temp = BaseConstraint.functional_constraints(
                        self.functions,
                        self.all_sensors,
                        self.data,
                        X_axis=self.X_axis,
                        Y_axis=self.Y_axis,
                        Field=self.Field,
                    )
                    G[:, i] = [x >= 0 for x in temp]
        else:
            G = np.zeros((len(self.all_sensors), 1), dtype=bool)
            if isinstance(self.data, np.ndarray):
                xValue, yValue = get_coordinates_from_indices(
                    self.all_sensors, self.data
                )
                for k in range(len(xValue)):
                    G[k, 0] = not eval(
                        self.equations[0], {"x": xValue[k], "y": yValue[k]}
                    )
            elif isinstance(self.data, pd.DataFrame):
                xValue, yValue = get_coordinates_from_indices(
                    self.all_sensors,
                    self.data,
                    X_axis=self.X_axis,
                    Y_axis=self.Y_axis,
                    Field=self.Field,
                )
                for k in range(len(xValue)):
                    G[k, 0] = not eval(
                        self.equations[0], {"x": xValue[k], "y": yValue[k]}
                    )
        idx_const, rank = BaseConstraint.get_functionalConstraind_sensors_indices(
            self.all_sensors, G[:, 0]
        )
        return idx_const, rank
