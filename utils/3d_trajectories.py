import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import numpy.linalg as linalg

def pca_transform(data, n_components=3):
    '''projects the data onto the first n principal components
    by eigenvalue decomposition of the covariance matrix
    '''
    
    eigvals, eigvecs = linalg.eigh(np.cov(data))
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:n_components]
    eigvecs = eigvecs[:,idx][:,:n_components]
    return np.dot(eigvecs.T, data)

data_folder = '../data/'
filenames = os.listdir(data_folder)

data = []

# no separation in segments, since then there
# would be only two neurons
for filename in filenames:
    filename = os.path.join(data_folder, filename)
    data.append(np.loadtxt(filename))

data = np.array(data)
data -= data.mean(axis=1)[:,np.newaxis]

transf_data = pca_transform(data)

# the SVD implementation from sklearn gives the
# same result, but with a flipped sign (because
# of the SVD implementation)

# from sklearn.decomposition import PCA as sklearnPCA
# sklearn_pca = sklearnPCA(n_components=3)
# sklearn_transf = sklearn_pca.fit_transform(data.T).T

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
transf_data = sklearn_pca.fit_transform(data.T).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(transf_data[0], transf_data[1], transf_data[2])
plt.savefig('../graphs/3d_trajectory.png')
plt.show()
