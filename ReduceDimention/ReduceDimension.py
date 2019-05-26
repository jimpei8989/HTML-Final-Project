import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from Utils.util import *

if __name__ == "__main__":
    dataDir = sys.argv[1]

    trainX = LoadData(os.path.join(dataDir,'X_train.npz'))
    print('-> Data Loaded', file = sys.stderr)

    reduced_dim = 20
    picked_feature_idx = np.linspace(0, 4999, reduced_dim+1,dtype=np.int)
    print(picked_feature_idx.shape)
    picked_feature = trainX[:, picked_feature_idx]
    thetas = np.arctan(np.diff(picked_feature) / np.diff(picked_feature_idx))
    print(thetas.shape)
    delta_thetas = np.concatenate((thetas[:, 0:1], np.diff(thetas)), axis=1)
    
    plt.rcParams['axes.unicode_minus']=False
    for t in delta_thetas[np.argpartition(np.max(np.abs(delta_thetas)[:,1:],axis=1),-50)[:50]]:
        plt.plot(np.arange(1,reduced_dim), t[1:]*180/np.pi*1000)
    plt.show()