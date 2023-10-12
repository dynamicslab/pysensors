
"""
Various utility functions for mapping constrained sensors locations with the column indices for class GQR.
"""

import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

def get_constrained_sensors_indices_linear(x_min, x_max, y_min, y_max,df,**kwgs):
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
    if 'X_axis' in kwgs.keys():
        X_axis = kwgs['X_axis']
    else:
        raise Exception('Must provide Y_axis as **kwgs as your data is a dataframe')
    if 'Y_axis' in kwgs.keys():
        Y_axis = kwgs['Y_axis']
    else:
        raise Exception('Must provide Y_axis as **kwgs as your data is a dataframe')
    x = df[X_axis].to_numpy()   ### Needs to be changed to get the X_axis and Y_axis value of what is in the user dataframe
    n_features = x.shape[0]
    y = df[Y_axis].to_numpy()
    idx_constrained = []
    for i in range(n_features):
        if (x[i] >= x_min and x[i] <= x_max) and (y[i] >= y_min and y[i] <= y_max):
            idx_constrained.append(i)
    return idx_constrained

class BaseConstraint(object):
    '''
    A General class for handling various functional and user-defined constraint shapes.
    It extends the ability of constraint handling with various plotting and annotating 
    functionalities while constraining various user-defined regions on the grid. 
    
    @ authors: Niharika Karnik (@nkarnik2999), Mohammad Abdo (@Jimmy-INL), and Joshua Cogliati (@joshua-cogliati-inl)
    '''
    def __init__(self,**kwgs):
        """
        Attributes
        ----------
        X_axis : string,
            Name of the column in dataframe to be plotted on the X axis.
        Y-axis : string,
            Name of the column in dataframe to be plotted on the Y axis.
        Field : string,
            Name of the column in dataframe to be plotted as a contour map.
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images) containing measurement data
        """
        if 'data' in kwgs.keys():
            self.data = kwgs['data']
        else:
            raise Exception('Must provide data as **kwgs')
        if isinstance(self.data,pd.DataFrame):
            if 'X_axis' in kwgs.keys():
                self.X_axis = kwgs['X_axis']
            else:
                raise Exception('Must provide X_axis as **kwgs as your data is a dataframe')
            if 'Y_axis' in kwgs.keys():
                self.Y_axis = kwgs['Y_axis']
            else:
                raise Exception('Must provide Y_axis as **kwgs as your data is a dataframe')
            if 'Field' in kwgs.keys():
                self.Field = kwgs['Field']
            else:
                raise Exception('Must provide Field as **kwgs as your data is a dataframe')
        
    
    def get_constraint_indices(self,all_sensors,info):
        '''
        A function for computing indices which lie within the region constrained by the user
        Attributes
        ----------
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer
        info : pandas.DataFrame/np.ndarray shape [n_features, n_samples],
            Dataframe or Matrix which represent the measurement data.
        Returns
        -----------
        idx_const : np.darray, shape [No. of constrained locations], 
            array which contains the constrained locations of the grid in terms of column indices of basis_matrix.
        rank : np.darray, shape [No. of constrained locations], 
            array which contains rank of the constrained sensor locations
        '''
        x, y = get_coordinates_from_indices(all_sensors,info)
        g = np.zeros(len(x),dtype = float)
        for i in range(len(x)):
             g[i] = self.constraint_function(x[i], y[i])
        G_const = constraints_eval([g],all_sensors,data = info)
        idx_const, rank = get_functionalConstraind_sensors_indices(all_sensors,np.squeeze(G_const))
        return idx_const,rank
    
    def draw_constraint(self):
        '''
        Function for drawing the constraint defined by the user
        '''
        fig , ax = plt.subplots()
        self.draw(ax)
        
    def plot_constraint_on_data(self,plot_type):
        '''
        Function for plotting the user-defined constraint on the data
        Attributes
        ----------
        data : pandas.DataFrame/np.darray [n_samples, n_features],
            dataframe (used for scatter and contour plots) or matrix (used for images) containing measurement data
        plot_type : string,
            the type of plot used to display the data
            image : if the data is represented in the fprm of an image
            scatter: if the data can be represented with a scatter plot
            contour_map: if the data can be represented in the form of a contour map
        Returns
        -----------
        A plot of the constraint on top of the measurement data plot.
        '''
        if plot_type == 'image': 
            image = self.data[1,:].reshape(1,-1)
            n_samples, n_features = self.data.shape
            image_shape = (int(np.sqrt(n_features)),int(np.sqrt(n_features)))
            fig , ax = plt.subplots()
            for i, comp in enumerate(image):
                vmax = max(comp.max(), -comp.min())
                ax.imshow(comp.reshape(image_shape), cmap = plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax )
        elif plot_type == 'scatter': 
            fig , ax = plt.subplots()
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            ax.scatter(x_vals, y_vals, color = 'blue', marker = '.')
        elif plot_type == 'contour_map': 
            fig , ax = plt.subplots()
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            ax.scatter(x_vals, y_vals, c = self.data[self.Field], cmap = plt.cm.coolwarm, s = 1)
        self.draw(ax)
        
    def plot_grid(self,all_sensors):
        '''
        Function to plot the grid with data points that signify sensor locations to choose from
        Attributes
        ----------
        all_sensors : np.darray,
            A ranked list of all sensor indices computed from just QR optimizer
        Returns
        -----------
        A plot of the user defined grid showing all possible sensor locations 
        '''
        if isinstance(self.data,np.ndarray):
            n_samples, n_features = self.data.shape
            info = (int(np.sqrt(n_features)),int(np.sqrt(n_features)))
            x_val, y_val = get_coordinates_from_indices(all_sensors,info)
            fig , ax = plt.subplots()
            ax.scatter(x_val, y_val, color = 'blue', marker = '.')
        elif isinstance(self.data,pd.DataFrame):
            y_vals = self.data[self.Y_axis]
            x_vals = self.data[self.X_axis]
            fig , ax = plt.subplots()
            ax.scatter(x_vals, y_vals, color = 'blue', marker = '.')
        
    def sensors_dataframe(self,sensors):
        '''
        Function to form a dataframe of the sensor index along with it's coordinate (X,Y,Z) positions 
        Attributes
        ----------
        sensors : np.darray,
            A ranked list of all sensor indices choosen from QR/CCQR/GQR optimizer
        Returns
        -----------
        A dataframe of the sensor locations choosen
        '''
        n_samples, n_features = self.data.shape
        n_sensors = len(sensors)
        if isinstance(self.data,np.ndarray):
            xTop = np.mod(sensors,np.sqrt(n_features))
            yTop = np.floor(sensors/np.sqrt(n_features))
        elif isinstance(self.data,pd.DataFrame):
            xTop, yTop = get_coordinates_from_indices(sensors,self.data)
        columns = ['Sensor ID','SensorX','sensorY'] 
        Sensors_df = pd.DataFrame(data = np.vstack([sensors,xTop,yTop]).T,columns=columns,dtype=float)
        Sensors_df.head(n_sensors)
        return Sensors_df
        
    def annotate_sensors(self,sensors,color='red'):
        '''
        Function to annotate the sensor location on the grid with the rand of the sensor
        Attributes
        ----------
        sensors : np.darray,
            A ranked list of all sensor indices choosen from QR/CCQR/GQR optimizer
        Returns
        -----------
        Annotation of sensor rank near the choosen sensor locations
        '''
        n_samples, n_features = self.data.shape
        n_sensors = len(sensors)
        if isinstance(self.data,np.ndarray):
            xTop = np.mod(sensors,np.sqrt(n_features))
            yTop = np.floor(sensors/np.sqrt(n_features))
            data = np.vstack([sensors,xTop,yTop]).T
            plt.plot(xTop, yTop,'*',color=color,alpha=0.3)
            for ind,i in enumerate(range(len(xTop))):
                plt.annotate(f"{str(ind)}",(xTop[i],yTop[i]),xycoords='data',
                    xytext=(-20,20), textcoords='offset points',color=color,fontsize=12,
                    arrowprops=dict(arrowstyle="->", color='black'))
        elif isinstance(self.data,pd.DataFrame):
            xTop, yTop = get_coordinates_from_indices(sensors,self.data)    #### Annotate not working for dataframe : FIX
            data = np.vstack([sensors,xTop,yTop]).T
            plt.plot(xTop, yTop,'*',color=color,alpha=0.3)
            for _,i in enumerate(range(len(sensors))):
                plt.annotate(f"{str(i)}",(xTop.to_numpy()[i]*100,yTop.to_numpy()[i]*100),xycoords='data',
                    xytext=(-20,20), textcoords='offset points',color=color,fontsize=12,
                    arrowprops=dict(arrowstyle="->", color='black'))
    
class Circle(BaseConstraint):
    '''
    General class for dealing with circular user defined constraints.
    Plotting, computing constraints functionalities included. 
    '''
    def __init__(self,center_x,center_y,radius,loc='in', **kwgs):
        super().__init__(**kwgs)
        '''
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
        '''
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.loc = loc
        
    def draw(self,ax):
        '''
        Function to plot a circle based on user-defined coordinates 
        Attributes
        ----------
        ax : axis on which the constraint circle should be plotted
        '''
        c = patches.Circle((self.center_x, self.center_y), self.radius, fill = False, color = 'r', lw = 2)
        ax.add_patch(c)
        ax.autoscale_view()
        
        
    def constraint_function(self,x, y):
        '''
        Function to compute whether a certain point on the grid lies inside/outside the defined constrained region 
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        '''
        if self.loc == 'in':
            return ((x-self.center_x)**2 + (y-self.center_y)**2) - self.radius**2
        else:
            return -(((x-self.center_x)**2 + (y-self.center_y)**2) - self.radius**2)
        
           
class Line(BaseConstraint):
    '''
    General class for dealing with linear user defined constraints.
    Plotting, computing constraints functionalities included. 
    '''
    def __init__(self,x1,x2,y1,y2,**kwgs):
        super().__init__(**kwgs)
        '''
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
        '''
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    def draw(self,ax):
        '''
        Function to plot a line based on user-defined coordinates 
        Attributes
        ----------
        ax : axis on which the constraint line should be plotted
        '''
        ax.plot([self.x1,self.x2],[self.y1,self.y2],'-r')
            
    def constraint_function(self,x, y):
        '''
        Function to compute whether a certain point on the grid lies inside/outside the defined constrained region 
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        '''
        return (y-self.y1)*(self.x2-self.x1) - (self.y2-self.y1)*(x-self.x1)
    
        
class Parabola(BaseConstraint):
    '''
    General class for dealing with parabolic user defined constraints.
    Plotting, computing constraints functionalities included.
    '''
    def __init__(self,h,k,a, **kwgs):
        super().__init__(**kwgs)
        '''
        Attributes
        ----------
        h : float,
            x-coordinate of the vertex of the parabola we want to be constrained
        k : float,
            y-coordinate of the vertex of the parabola we want to be constrained
        a : float,
            x-coordinate of the focus of the parabola
        '''
        self.h = h
        self.k = k
        self.a = a
        
class Ellipse(BaseConstraint):
    '''
    General class for dealing with elliptical user defined constraints.
    Plotting, computing constraints functionalities included. 
    '''
    def __init__(self,center_x,center_y,half_major_axis, half_minor_axis,angle=0.0,loc='in', **kwgs):
        super().__init__(**kwgs)
        '''
        Attributes
        ----------
        center_x : float,
            x-coordinate of the center of circle
        center_y : float,
            y-coordinate of the center of circle
        half_major_axis : float,
            half the length of the major axis
        half_minor_axis : float,
            half the length of the minor axis
        angle : float,
            angle of the orientation of the ellipse in degrees
        loc : string- 'in'/'out',
            specifying whether the inside or outside of the shape is constrained
        '''
        self.center_x = center_x
        self.center_y = center_y
        self.half_major_axis = half_major_axis
        self.half_minor_axis = half_minor_axis
        self.loc = loc
        self.angle = angle
        
    def draw(self,ax):
        '''
        Function to plot an ellipse based on user-defined coordinates 
        Attributes
        ----------
        ax : axis on which the constraint line should be plotted
        '''
        if self.angle % 360.0 in [0,180]:
        # if self.half_major_axis > self.half_minor_axis:
            c = patches.Ellipse((self.center_x, self.center_y), self.half_major_axis, self.half_minor_axis, angle=self.angle, fill = False, color = 'r', lw = 2)
        elif self.angle % 270 in [0,90]: 
            c = patches.Ellipse((self.center_x, self.center_y), self.half_minor_axis, self.half_major_axis, fill = False, color = 'r', lw = 2)
        else:
            c = patches.Ellipse((self.center_x, self.center_y), self.half_minor_axis, self.half_major_axis, angle=self.angle, fill = False, color = 'r', lw = 2)
        ax.add_patch(c)
        ax.autoscale_view()
        
        
    def constraint_function(self,x, y):
        '''
        Function to compute whether a certain point on the grid lies inside/outside the defined constrained region 
        Attributes
        ----------
        x : float,
            x coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        y : float,
            y coordinate of point on the grid being evaluated to check whether it lies inside or outside the constrained region
        '''
        if self.loc == 'in':
            return (((x-self.center_x)**2)*(self.half_minor_axis**2) + ((y-self.center_y)**2)*(self.half_major_axis**2)) - (self.half_major_axis**2 * self.half_minor_axis**2)
        else: 
            return - ((((x-self.center_x)**2)*(self.half_minor_axis**2) + ((y-self.center_y)**2)*(self.half_major_axis**2)) - (self.half_major_axis**2 * self.half_minor_axis**2))

class Polygon(BaseConstraint):
    '''
    General class for dealing with polygonal user defined constraints.
    Plotting, computing constraints functionalities included. 
    '''
    def __init__(self,xy_coords, **kwgs):
        super().__init__(**kwgs)
        '''
        Attributes
        ----------
        xy_coords : (N,2) array_like,
            an array consisting of tuples for (x,y) coordinates of points of the Polygon where N = No. of sides of the polygon
        '''
        self.xy_coords= xy_coords
        
    def draw(self,ax):
        c = patches.Polygon(self.xy_coords, fill = False, color = 'r', lw = 2)
        ax.add_patch(c)
        ax.autoscale_view()
        
        
    def constraint_function(self,x_all_unc, y_all_unc):
        '''
        To be Filled
        '''
        # def point_in_polygon(point, polygon):
        x, y = x_all_unc,y_all_unc
        polygon =self.xy_coords 
        n = len(polygon)
        inside = False

        # x1, y1 = polygon[0]
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            if y1 < y and y2 >= y or y2 < y and y1 >= y:
                if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                    inside = not inside

        return not inside
        # for i in range(n + 1):
        #     x2, y2 = polygon[i % n]
        #     if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
        #         if x1 != x2:
        #             x_intersection = (y - y1) * (x2 - x1) / (y2 - y1) + x1
        #             if x1 == x2 or x <= x_intersection:
        #                 inside = not inside
        #     x1, y1 = x2, y2

        # return inside

# # Example usage:
# polygon = [(0, 0), (0, 4), (4, 4), (4, 0)]
# point = (2, 2)
# result = point_in_polygon(point, polygon)
# print(result)  # True (point is inside the polygon)

        # return ((x_all_unc-self.center_x)**2 + (y_all_unc-self.center_y)**2) - self.radius**2
class UserDefinedConstraints(BaseConstraint):
    '''
    General class for dealing with any form of user defined constraints.
    Plotting, computing constraints functionalities included. 
    '''
    def __init__(self,all_sensors,info,const_path, **kwgs):
        super().__init__(**kwgs)
        '''
        Attributes
        ----------
        const_path : 'string',
            path of the file that constains the user-defined constraint function
        '''
        self.all_sensors = all_sensors
        self.info = info
        self.const_path = const_path
        
    def draw(self,ax):
        '''
        To be filled
        '''
        nConstraints = len([self.const_path])
        G = np.zeros((len(self.all_sensors),nConstraints),dtype=bool)
        for i in range(nConstraints):
            if isinstance(self.info,tuple):
                temp = functional_constraints([self.const_path][i],self.all_sensors,shape = self.info)
                G[:,i] = [x != 0 for x in temp]
            elif isinstance(self.info,pd.DataFrame):
                temp = functional_constraints([self.const_path][i],self.all_sensors,data = self.info)
                G[:,i] = [x != 0 for x in temp]
        idx_const, rank = get_functionalConstraind_sensors_indices(self.all_sensors,G[:,0]) ## Now I am getting the indices where the function has a value == 0
        x_val,y_val = get_coordinates_from_indices(idx_const,self.info)
        ax.scatter(x_val,y_val,s = 5 )
         
    def constraint(self):
        '''
        To be Filled
        '''
        nConstraints = len([self.const_path])
        G = np.zeros((len(self.all_sensors),nConstraints),dtype=bool)
        for i in range(nConstraints):
            if isinstance(self.info,tuple):
                temp = functional_constraints([self.const_path][i],self.all_sensors,shape = self.info)
                G[:,i] = [x>=0 for x in temp]
            elif isinstance(self.info,pd.DataFrame):
                temp = functional_constraints([self.const_path][i],self.all_sensors,data = self.info)
                G[:,i] = [x>=0 for x in temp]
        idx_const, rank = get_functionalConstraind_sensors_indices(self.all_sensors,G[:,0])
        return idx_const,rank
        
def functional_constraints(functionHandler, idx,**kwargs):
        """
        Function for evaluating the functional constraints.

        Parameters
        ----------
        functionHandler : function, a function handle to the function which is to be evaluated
        idx : np.ndarray, ranked list of sensor locations (column indices)
        shape : tuple of ints, Shape of the matrix fed as data to the algorithm
        data : pandas.DataFrame, Dataframe which represents the measurement data.

        Return
        ------
        g : function, Contains the function defined by the user for the functional constraint. 
        """
        if 'shape' in kwargs.keys():
            shape = kwargs['shape']
            xLoc,yLoc = get_coordinates_from_indices(idx,shape)
        elif 'data' in kwargs.keys():
            data = kwargs['data']
            ## TODO:
            #xLoc = data.loc[idx, 'X (m)']
            #yLoc = data.loc[idx, 'Y (m)']
            xLoc,yLoc =  get_indices_from_dataframe(idx,data)
        functionName = os.path.basename(functionHandler).strip('.py')
        dirName = os.path.dirname(functionHandler)
        sys.path.insert(0,os.path.expanduser(dirName))
        module = __import__(functionName)
        func = getattr(module, functionName)
        g = func(xLoc, yLoc,**kwargs)
        return g
    
def constraints_eval(constraints,senID,**kwargs):
    """
    Function for evaluating whether a certain sensor index lies within the constrained region or not.

    Parameters:
    ---------- 
        constraints: __(type?)__, The constraint defined by the user 
        senID: np.ndarray, shape [n_features], ranked list of sensor locations (column indices)
        data : pandas.DataFrame/np.ndarray shape [n_features, n_samples]
                Dataframe or Matrix which represent the measurement data.
    Returns
    -------
    G : Boolean np.darray, shape [n_features], array which contains a Boolean value based on whether a column index is constrained or not.
    """
    nConstraints = len(constraints)
    G = np.zeros((len(senID),nConstraints),dtype=bool)
    for i in range(nConstraints):
        # temp = BaseConstraint.functional_constraints(constraints[i],senID,kwargs)
        G[:,i] = [x>0 for x in constraints[i]]
    return G
    
def get_functionalConstraind_sensors_indices(senID,g):
    """
    Function for finding constrained sensor locations on the grid and their ranks

    Parameters
    ----------
    senID: np.darray, ranked list of sensor locations (column indices)
    g : float, constraint evaluation function (negative if violating the constraint)

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix.
    rank : np.darray, shape [No. of constrained locations], array which contains rank of the constrained sensor locations
    """
    assert (len(senID)==len(g))
    idx_constrained = senID[~g].tolist()
    rank = np.where(np.isin(senID,idx_constrained))[0].tolist() # ==False
    return idx_constrained, rank
    
def order_constrained_sensors(idx_constrained_list, ranks_list):
    """
    Function for ordering constrained sensor locations on the grid according to their ranks.

    Parameters
    ----------
    idx_constrained_list : np.darray shape [No. of constrained locations], Constrained sensor locations
    ranks_list : no.darray shape [No. of constrained locations], Ranks of each constrained sensor location

    Returns
    -------
    sortedConstraints : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix sorted according to their rank.
    ranks : np.darray, shape [No. of constrained locations], array which contains the ranks of constrained sensors. 
    """
    sortedConstraints,ranks =zip(*[[x,y] for x,y in sorted(zip(idx_constrained_list, ranks_list),key=lambda x: (x[1]))])
    return sortedConstraints,ranks
    
def get_coordinates_from_indices(idx,info):
    """
    Function for obtaining the coordinates on a grid from column indices

    Parameters
    ----------
    idx :  int, sensor ID
    info : pandas.DataFrame/np.ndarray shape [n_features, n_samples], Dataframe or Matrix which represent the measurement data.

    Returns
    -------
    idx_constrained : np.darray, shape [No. of constrained locations], array which contains the constrained
        locations of the grid in terms of column indices of basis_matrix.

    Returns:
        (x,y) : tuple, The coordinates on the grid of each sensor. 
    """
    if isinstance(info,tuple):
        return np.unravel_index(idx,info,'F')
    elif isinstance(info,pd.DataFrame):
        x = info.loc[idx,'X (m)']#.values  ## This will not always be X (m) or Y (m) CHANGE!!!
        y = info.loc[idx,'Y (m)']#.values
        return (x,y)
    
def get_indices_from_coordinates(coordinates,shape):
    """
    Function for obtaining the indices of columns/sensors from coordinates on a grid when data is in the form of a matrix
    
    Parameters
    ----------
    coordinates : tuple of array_like , (x,y) pair coordinates of sensor locations on the grid
    shape : tuple of ints, Shape of the matrix fed as data to the algorithm

    Returns
    -------
    np.ravel_multi_index(coordinates,shape,order='F') : np.ndarray, The indices of the sensors. 
    """
    return np.ravel_multi_index(coordinates,shape,order='F')

def get_indices_from_dataframe(idx,df): ## Niharikas_comment : I think this should be renamed to get coordinates from dataframe as when given a sensor index it returns a tuple containing the coordinates of that sensor index.
    ## It can also maybe be removed completely as get_coordinates_from_indices(idx,info) does the same thing. Thoughts? 
    
    x = df['X (m)'].to_numpy()
    y = df['Y (m)'].to_numpy()
    return(x[idx],y[idx])

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