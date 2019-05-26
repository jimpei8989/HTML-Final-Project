import numpy as np
from sklearn.utils import shuffle

lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

# Reference: https://stackoverflow.com/questions/18231135/load-compressed-data-npz-from-file-using-numpy-load
def LoadData(path):
    with np.load(path) as npf:
        a = npf.f.arr_0
    return a

# Given data directory, return X_train, Y_train, X_test
def LoadAll(path = 'data/'):
    path = path if path[-1] == '/' else path + '/'
    return LoadData(path + 'X_train.npz'), LoadData(path + 'Y_train.npz'), LoadData(path + 'X_test.npz')

def DataSplit(X, Y):
    Xp, Yp = shuffle(X, Y, random_state = lucky_num)
    cut = int(Xp.shape[0] * 0.8)
    return Xp[:cut, :], Yp[:cut, :], Xp[cut:, :], Yp[:cut, :]

class Model:
    def __init__(self, *args):
        pass

    def fit(self, trainX, trainY, validX, validY, *args):
        '''
        Args
            trainX: 2D array (num * input_dim)
            trainY: 2D array (num * output_dim)
            validX: 2D array (num * input_dim)
            validY: 2D array (num * output_dim)
        Returns
            List of tuples, indicating the three
        '''
        raise NotImplementedError

    def score(self, X, Y, *args):
        '''
        Args
            X: 2D array (num * input_dim)
            Y: 2D array (num * output_dim)
        Returns
            3-tuple (MSE, WMAE, NAE)
        '''
        raise NotImplementedError

    def predict(self, X, *args):
        '''
        Args
            X: 2D array (num * input_dim)
        Returns
            Y: 2D array (num * output_dim)
        '''
        raise NotImplementedError

    def load_model(self, directory_path):
        '''
        Description
            Load models from given path
        Args
            directory_path: a directory ending with '/'
        Returns
            None
        '''

    def save_model(self, directory_path):
        '''
        Description
            Save models to given path
        Args
            directory_path: a directory ending with '/'
        Returns
            None
        '''

