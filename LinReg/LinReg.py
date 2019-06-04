import os, sys, pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from Utils.util import *
from Utils.ReduceDimension import reduce_dimension

class LinReg(Model):
    def __init__(self):
        super().__init__()

    def _fit(self, trainX, trainY):
        self.reg = LinearRegression(normalize=True, n_jobs=os.cpu_count()).fit(trainX, trainY)
        return self

    def _predict(self, X):
        return self.reg.predict(X)

if __name__ == "__main__":
    # The lucky number is the sha256sum of our team name!
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

    data_dir = sys.argv[1]
    model_path = sys.argv[2]

    #trainX, trainY = load_data(data_dir + '/X_train.npz'),load_data(data_dir+'/Y_train.npz')
    trainX, trainY, testX = LoadAll(data_dir)
    # testX = LoadData(data_dir + '/X_test.npz')
    print('-> Data Loaded', file=sys.stderr)

    try:
        reg = load_model(model_path)
        print('Load Model')
    except FileNotFoundError:
        pca=load_model(os.path.join(data_dir,'pca_model100'))
        print('PCA loaded')
        reg = LinReg().fit(trainX, trainY, transform_args = pca)
        save_model(reg, model_path)

    #print('Training Score:', reg.score(trainX, trainY))

    generate_csv(reg,testX)
