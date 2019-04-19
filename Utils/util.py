import numpy as np

# Reference: https://stackoverflow.com/questions/18231135/load-compressed-data-npz-from-file-using-numpy-load
def LoadData(path):
    with np.load(path) as npf:
        a = npf.f.arr_0
    return a

# Given data directory, return X_train, Y_train, X_test
def LoadAll(path = 'data/'):
    path = path if path[-1] == '/' else path + '/'
    return LoadData(path + 'X_train.npz'), LoadData(path + 'Y_train.npz'), LoadData(path + 'X_test.npz')

