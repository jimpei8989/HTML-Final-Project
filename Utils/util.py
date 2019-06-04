import pickle, os
import numpy as np
from sklearn.utils import shuffle

lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

# Reference: https://stackoverflow.com/questions/18231135/load-compressed-data-npz-from-file-using-numpy-load
def LoadData(path):
    '''
    Desciption:
        Load one file from given path. It will try to load .npy first, and if .npy does not exist, it will load .npz file and save .npy file in path.
    Args
        path: model path
    Returns
        model
    '''
    if not os.path.exists(path[:-1]+'y'):
        with np.load(path[:-1]+'z') as npf:
            a = npf.f.arr_0
        np.save(path[:-1]+'y',a)
        return a
    else:
        return np.load(path[:-1]+'y')

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
        pickle.dump(model, f)

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

class FeatureExtraction:
    def __init__(self, func=None):
        self.func = func
    def fit(self, X, *args):
        return self
    def transform(self, X, *args):
        return self.func(X, *args)

class Model:
    def __init__(self, *args):
        self.fitted = False

    def _fit(self, trainX, trainY, *args, validX=None, validY=None):
        '''
        Args
            trainX: 2D array (num * input_dim)
            trainY: 2D array (num * output_dim)
            validX: 2D array (num * input_dim)
            validY: 2D array (num * output_dim)
            args:   other argument
        Returns
            self
        '''
        raise NotImplementedError

    def fit(self, trainX, trainY, *args, transform_args=(), validX=None, validY=None):
        '''
        Args
            trainX: 2D array (num * input_dim)
            trainY: 2D array (num * output_dim)
            validX: 2D array (num * input_dim)
            validY: 2D array (num * output_dim)
            args:   other arguments passed to _fit function
            transform_args: tuple of the feature extracting function(object) and its arguments
        Returns
            self
        '''
        if transform_args:
            if not isinstance(transform_args, tuple):
                transform_args = (transform_args,) # convert to a tuple
            if callable(transform_args[0]): # it is a function
                self.featex = FeatureExtraction(transform_args[0])
            else:
                self.featex = transform_args[0]

            if not hasattr(self.featex, 'fitted') or not self.featex.fitted:
                self.featex.fit(trainX, *transform_args[1:])
            trainX = self.featex.transform(trainX,*transform_args[1:])
        
        self.fitted = True
        if validX is not None and validY is not None:
            return self._fit(trainX, trainY, *args, validX=validX, validY=validY)
        else:
            return self._fit(trainX, trainY, *args)

    def score(self, X, Y, *args, transform_args = ()):
        '''
        Args
            X: 2D array (num * input_dim)
            Y: 2D array (num * output_dim)
        Returns
            2-tuple (WMAE, NAE)
        '''
        w=np.array([[300],[1],[200]])
        abs_value=np.abs(self.predict(X, *args, transform_args = transform_args)-Y)
        return np.mean(abs_value @ w), np.sum(abs_value / Y) / X.shape[0]

    def _predict(self, X, *args):
        '''
        Args
            X: 2D array (num * input_dim)
            args:   other arguments
        Returns
            Y: 2D array (num * output_dim)
        '''
        raise NotImplementedError

    def predict(self, X, *args, transform_args = ()):
        '''
        Args
            X: 2D array (num * input_dim)
            args:   other arguments passed to _predict function
            transform_args: tuple of the feature extracting function(object) and its arguments
        Returns
            Y: 2D array (num * output_dim)
        '''
        if hasattr(self, 'fitted') and not self.fitted:
            raise Exception('The model has not been fitted yet!')
        if hasattr(self, 'featex'):
            X = self.featex.transform(X, *transform_args)
        return self._predict(X, *args)
        
