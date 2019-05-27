import pickle
import numpy as np
from sklearn.utils import shuffle

lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

# Reference: https://stackoverflow.com/questions/18231135/load-compressed-data-npz-from-file-using-numpy-load
def LoadData(path):
    with np.load(path) as npf:
        a = npf.f.arr_0
    return a

def load_data(path):
    return LoadData(path)

# Given data directory, return X_train, Y_train, X_test
def LoadAll(path = 'data/'):
    path = path if path[-1] == '/' else path + '/'
    return LoadData(path + 'X_train.npz'), LoadData(path + 'Y_train.npz'), LoadData(path + 'X_test.npz')

def load_all(path='data/'):
    return LoadAll(path)


def load_model(path):
    '''
    Description
        Load models from given path
    Args
        path: model path
    Returns
        model
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, path):
    '''
    Description
        Save models to given path
    Args
        path: model path
    Returns
        None
    '''
    with open(path,'wb') as f:
        pickle.dump(model, path)

def generate_csv(path_or_clf,X,save_path='./output.csv'):
    '''
    Args
        path_or_clf: the path of model or model itself
        X: 2D array
        save_path: path to csv (default ./output.csv)
    Return
        None
    '''
    model=path_or_clf
    if isinstance(path_or_clf,str):
        model=load_model(path_or_clf)
    np.savetxt(save_path,model.predict(X),delimiter=',')

def DataSplit(X, Y):
    Xp, Yp = shuffle(X, Y, random_state = lucky_num)
    cut = int(Xp.shape[0] * 0.8)
    return Xp[:cut, :], Yp[:cut, :], Xp[cut:, :], Yp[:cut, :]

def data_split(X,Y):
    return DataSplit(X,Y)

class Model:
    def __init__(self, *args):
        pass

    def fit(self, trainX, trainY, validX=None, validY=None, *args):
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
            2-tuple (WMAE, NAE)
        '''
        w=np.array([[300],[1],[200]])
        return np.mean(np.abs(self.predict(X)-Y) @ w), np.sum(np.abs(self.predict(X)-Y) / Y) / X.shape[0]

    def predict(self, X, *args):
        '''
        Args
            X: 2D array (num * input_dim)
        Returns
            Y: 2D array (num * output_dim)
        '''
        raise NotImplementedError

        
