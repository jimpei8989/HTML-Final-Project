import os, sys, pickle
from multiprocessing import Pool
import numpy as np
from sklearn.linear_model import SGDRegressor

from Utils.util import *

class SGDReg(Model):
    def __init__(self):
        super().__init__()
        self.poo=Pool(3)
    
    def get_reg(self,tup):
        idx=np.random.shuffle(np.arange(tup[0].shape[0]))
        X, Y = tup[0][idx], tup[1][idx]
        return SGDRegressor(loss='epsilon_insensitive', penalty = 'none', epsilon = 0, shuffle = False, max_iter = 1000, average = 32).fit(X, Y)

    def fit(self, trainX, trainY,*args):
        self.regs=self.poo.map(self.get_reg, [(trainX.copy(), trainY[:,i].copy()) for i in range(3)])
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
