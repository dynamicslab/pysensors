#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pysensors as ps
# from pysensors.optimizers._ccqr import CCQR


# In[2]:


faces = datasets.fetch_olivetti_faces(shuffle=True)
X = faces.data

n_samples, n_features = X.shape
print('Number of samples:', n_samples)
print('Number of features (sensors):', n_features)

# Global centering
X = X - X.mean(axis=0)

# Local centering
X -= X.mean(axis=1).reshape(n_samples, -1)


# In[3]:


n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)

def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    '''Function for plotting faces'''
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()


# In[4]:


plot_gallery("First few centered faces", X[:n_components])


# In[5]:
# reduce the X
imageSize = 64
image_shape = (imageSize, imageSize)

X = X[:,:imageSize**2]
n_features = X.shape[1]

#Find all sensor locations using built in QR optimizer
max_const_sensors = 230
n_const_sensors = 2
n_sensors = 20
optimizer  = ps.optimizers.QR()
model = ps.SSPOR(optimizer=optimizer, n_sensors=n_sensors)
model.fit(X)

all_sensors = model.get_all_sensors()
print(all_sensors)


# In[6]:


#Define Constrained indices
a = np.unravel_index(all_sensors, (imageSize,imageSize))
print(a)
a_array = np.transpose(a)
print(a_array.shape)
#idx = np.ravel_multi_index(a, (64,64))
#print(idx)
xmin = 0
xmax = 10
ymin = 40
ymax = 64

constrained_sensorsx = []
constrained_sensorsy = []
for i in range(n_features):
    if a[0][i] < xmax and a[1][i] > ymin:  # x<10 and y>40
        constrained_sensorsx.append(a[0][i])
        constrained_sensorsy.append(a[1][i])

constrained_sensorsx = np.array(constrained_sensorsx)
constrained_sensorsy = np.array(constrained_sensorsy)

constrained_sensors_array = np.stack((constrained_sensorsy, constrained_sensorsx), axis=1)
constrained_sensors_tuple = np.transpose(constrained_sensors_array)


#print(constrained_sensors_tuple)
#print(len(constrained_sensors_tuple))
idx_constrained = np.ravel_multi_index(constrained_sensors_tuple, (imageSize,imageSize))

#print(len(idx_constrained))
#print(constrained_sensorsx)
#print(constrained_sensorsy)
#print(idx_constrained)
print(np.sort(idx_constrained[:]))
all_sorted = np.sort(all_sensors)
#print(all_sorted)
idx = np.arange(all_sorted.shape[0])
#all_sorted[idx_constrained] = 0


# In[7]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

ax = plt.subplot()
#Plot constrained space
img = np.zeros(n_features)
img[idx_constrained] = 1
im = plt.imshow(img.reshape(image_shape),cmap=plt.cm.binary)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
plt.title('Constrained region')
plt.show()


# In[8]:


#New class for constrained sensor placement
from pysensors.optimizers._qr import QR
class GQR(QR):
    """
    General QR optimizer for sensor selection.
    Ranks sensors in descending order of "importance" based on
    reconstruction performance.

    See the following reference for more information

        Manohar, Krithika, et al.
        "Data-driven sparse sensor placement for reconstruction:
        Demonstrating the benefits of exploiting known patterns."
        IEEE Control Systems Magazine 38.3 (2018): 63-86.
    """
    def __init__(self):
        """
        Attributes
        ----------
        pivots_ : np.ndarray, shape [n_features]
            Ranked list of sensor locations.
        """
        self.pivots_ = None

    def fit(
        self,
        basis_matrix, idx_constrained, const_sensors
    ):
        """
        Parameters
        ----------
        basis_matrix: np.ndarray, shape [n_features, n_samples]
            Matrix whose columns are the basis vectors in which to
            represent the measurement data.
        optimizer_kws: dictionary, optional
            Keyword arguments to be passed to the qr method.

        Returns
        -------
        self: a fitted :class:`pysensors.optimizers.CCQR` instance
        """

        n, m = basis_matrix.shape  # We transpose basis_matrix below

        ## Assertions and checks:
        if n_sensors > n_features - max_const_sensors + n_const_sensors: ##TODO should be moved to the class
            raise IOError ("n_sensors cannot be larger than n_features - all possible locations in the constrained area + allowed constrained sensors")
        if n_sensors > n_samples + n_const_sensors:
            raise IOError ("Currently n_sensors should be less than number of samples + number of constrained sensors,\
                           got: n_sensors = {}, n_samples + n_const_sensors = {} + {} = {}".format(n_sensors,n_samples,n_const_sensors,n_samples+n_const_sensors))

        # Initialize helper variables
        R = basis_matrix.conj().T.copy()
        #print(R.shape)
        p = np.arange(n)
        #print(p)
        k = min(m, n)


        for j in range(k):
            r = R[j:, j:]
            # Norm of each column
            dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))

            # if j < n_const_sensors:
            dlens_updated = f_region(idx_constrained,dlens,p,j, const_sensors)
            # else:
                # dlens_updated = dlens
            # Choose pivot
            i_piv = np.argmax(dlens_updated)
            #print(i_piv)


            dlen = dlens_updated[i_piv]

            if dlen > 0:
                u = r[:, i_piv] / dlen
                u[0] += np.sign(u[0]) + (u[0] == 0)
                u /= np.sqrt(abs(u[0]))
            else:
                u = r[:, i_piv]
                u[0] = np.sqrt(2)

            # Track column pivots
            i_piv += j # true permutation index is i_piv shifted by the iteration counter j
            print(i_piv) # Niharika's debugging line
            p[[j, i_piv]] = p[[i_piv, j]]
            print(p)


            # Switch columns
            R[:, [j, i_piv]] = R[:, [i_piv, j]]

            # Apply reflector
            R[j:, j:] -= np.outer(u, np.dot(u, R[j:, j:]))
            R[j + 1 :, j] = 0


        self.pivots_ = p


        return self
#function for mapping sensor locations with constraints
def f_region(lin_idx, dlens, piv, j, const_sensors):
    #num_sensors should be fixed for each custom constraint (for now)
    #num_sensors must be <= size of constraint region
    """
    Function for mapping constrained sensor locations with the QR procedure.

    Parameters
        ----------
        lin_idx: np.ndarray, shape [No. of constrained locations]
            Array which contains the constrained locations mapped on the grid.
        dlens: np.ndarray, shape [Variable based on j]
            Array which contains the norm of columns of basis matrix.
         num_sensors: int,
            Number of sensors to be placed in the constrained area.
        j: int,
            Iterative variable in the QR algorithm.

        Returns
        -------
        dlens : np.darray, shape [Variable based on j] with constraints mapped into it.
    """
    if j < const_sensors: # force sensors into constraint region
        #idx = np.arange(dlens.shape[0])
        #dlens[np.delete(idx, lin_idx)] = 0

        didx = np.isin(piv[j:],lin_idx,invert=True)
        dlens[didx] = 0
    else:
        didx = np.isin(piv[j:],lin_idx,invert=False)
        dlens[didx] = 0
    return dlens


# In[9]:



optimizer1 = GQR()
model1 = ps.SSPOR(optimizer = optimizer1, n_sensors = n_sensors)
model1.fit(X, quiet=True, prefit_basis=False, seed=None, idx_constrained = idx_constrained, const_sensors = n_const_sensors)


# In[10]:


all_sensors1 = model1.get_all_sensors()
print(all_sensors1[:n_const_sensors])

print(np.array_equal(np.sort(all_sensors),np.sort(all_sensors1)))

# In[12]:


top_sensors = model1.get_selected_sensors()
print(top_sensors)
## TODO: this can be done using ravel and unravel more elegantly
yConstrained = np.floor(top_sensors[:n_const_sensors]/np.sqrt(n_features))
xConstrained = np.mod(top_sensors[:n_const_sensors],np.sqrt(n_features))

img = np.zeros(n_features)
img[top_sensors[n_const_sensors:]] = 16
plt.plot(xConstrained,yConstrained,'*r')
plt.plot([xmin,xmin],[ymin,ymax],'r')
plt.plot([xmin,xmax],[ymax,ymax],'r')
plt.plot([xmax,xmax],[ymin,ymax],'r')
plt.plot([xmin,xmax],[ymin,ymin],'r')
plt.imshow(img.reshape(image_shape),cmap=plt.cm.binary)
plt.title('n_sensors = {}, n_constr_sensors = {}'.format(n_sensors,n_const_sensors))
plt.show()



# In[13]:


print(n_sensors)
print(idx_constrained.shape)


# In[ ]:




