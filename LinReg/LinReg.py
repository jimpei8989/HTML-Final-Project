import os, sys, pickle
import numpy as np
from sklearn.linear_model import LinearRegression

from Utils.util import *
from Utils.ReduceDimension import reduce_dimension, pca

class LinReg(Model):
    def __init__(self):
        super().__init__()
        self.fitted=False
        self.func = None

    def fit(self, trainX, trainY, *args):
        if args:
            self.func = args[0]
            trainX = self.func(trainX,*args[1:])
        self.reg = LinearRegression(normalize=True, n_jobs=os.cpu_count()).fit(trainX, trainY)
        self.fitted = True
        return self

    def predict(self, X, *args):
        if not self.fitted:
            raise Exception('The model has not been fitted yet!')
        return self.reg.predict(X if self.func is None else self.func(X,*args))

if __name__ == "__main__":
    # The lucky number is the sha256sum of our team name!
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

    data_dir = sys.argv[1]
    model_path = sys.argv[2]

    trainX, trainY = load_data(data_dir + '/X_train.npz'),load_data(data_dir+'/Y_train.npz')
    #trainX, trainY, testX = LoadAll(data_dir)
    # testX = LoadData(data_dir + '/X_test.npz')
    print('-> Data Loaded', file=sys.stderr)

    try:
        reg = load_model(model_path)
        print('Load Model')
    except FileNotFoundError:
        reg = LinReg()
        reg.fit(trainX, trainY,pca,reg)
        with open(model_path, 'wb') as f:
            pickle.dump(reg, f)

    print('Training Score:', reg.score(trainX, trainY))

    #generate_csv(reg,testX)
