import os, sys, pickle
from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import SGDRegressor

from Utils.util import *

class SGDReg(Model):
    @staticmethod
    def get_reg(X,Y):
        return SGDRegressor(loss='epsilon_insensitive', penalty = 'none', epsilon = 0, shuffle = False, max_iter = 10000, learning_rate = 'adaptive', average = 32).fit(X, Y)

    def fit(self, trainX, trainY,*args):
        idx=np.arange(trainX.shape[0])
        np.random.shuffle(idx)
        trainX, trainY = trainX[idx], trainY[idx]
        
        self.regs=Parallel(n_jobs=3, backend="threading")(delayed(SGDReg.get_reg)(trainX, trainY[:,i]) for i in range(trainY.shape[1]))
        return self

    def predict(self, X, *args):
        return np.concatenate([reg.predict(X).reshape(-1,1) for reg in self.regs], axis=1)

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
        reg = SGDReg().fit(trainX, trainY)
        with open(model_path, 'wb') as f:
            pickle.dump(reg, f)

    print('Training Score:', reg.score(trainX, trainY))
