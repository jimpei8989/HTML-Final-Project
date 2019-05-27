import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from Utils.util import *

def reduce_dimension(trainX,reduced_dim):
    delta_thetas = np.diff(np.arctan(np.diff(trainX[:,:5000])))
    delta_thetas = np.partition(delta_thetas,range(-reduced_dim,0))[:,-reduced_dim:] #partial sort in each row
    
    delta_thetas_draw = delta_thetas[np.argsort(np.max(delta_thetas,axis=1))[::-1]] #sort by the maximum value of all Xi
    plt.rcParams['axes.unicode_minus']=False
    for t in delta_thetas_draw[:50]:
        plt.scatter(np.arange(reduced_dim), t*180/np.pi*1000)
    plt.show()

    return np.concatenate((delta_thetas, trainX[:,5000:]), axis=1)

if __name__ == "__main__":
    dataDir = sys.argv[1]

    trainX = LoadData(os.path.join(dataDir,'X_train.npz'))
    print('-> Data Loaded', file = sys.stderr)

    reduce_dimension(trainX,20)
