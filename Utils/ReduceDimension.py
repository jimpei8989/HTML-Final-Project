import os, sys, pickle
import numpy as np
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from Utils.util import *

def reduce_dimension(trainX,reduced_dim):
    delta_thetas = np.diff(np.arctan(np.diff(trainX[:,:5000])))
    delta_thetas = np.partition(delta_thetas,range(-reduced_dim,0))[:,-reduced_dim:] #partial sort in each row
    
    #delta_thetas_draw = delta_thetas[np.argsort(np.max(delta_thetas,axis=1))[::-1]] #sort by the maximum value of all Xi
    #plt.rcParams['axes.unicode_minus']=False
    #for t in delta_thetas_draw[:50]:
    #    plt.scatter(np.arange(reduced_dim), t*180/np.pi*1000)
    #plt.show()

    return np.concatenate((delta_thetas, trainX[:,5000:]), axis=1)

def pca(X):
    pca_model = PCA(n_components = 100).fit(X)
    pca_model.fitted = True
    return pca_model

if __name__ == "__main__":
    dataDir = sys.argv[1]

    trainX = LoadData(os.path.join(dataDir,'X_train.npz'))
    print('-> Data Loaded', file = sys.stderr)

    #reduce_dimension(trainX,20)
    with open(os.path.join(dataDir,'pca_model100'),'wb') as f:
        pickle.dump(pca(trainX),f)
