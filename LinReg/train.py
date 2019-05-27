import os, sys, pickle
import numpy as np
from sklearn.linear_model import LinearRegression

from Utils.util import *

if __name__ == "__main__":
    # The lucky number is the sha256sum of our team name!
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

    dataDir = sys.argv[1]
    modelPath = sys.argv[2]

    # Xtrain, Ytrain, Xtest = LoadAll(dataDir)
    Xtest = LoadData(dataDir + '/X_test.npz')
    print('-> Data Loaded', file=sys.stderr)

    try:
        reg = load_model(modelPath)
        print('Load Model')
    except FileNotFoundError:
        Xtrain=Xtrain[:, [0]+list(range(5000,Xtrain.shape[1]))]
        reg = LinearRegression(normalize = True, n_jobs = 32)
        reg.fit(Xtrain, Ytrain)
        with open(modelPath, 'wb') as f:
            pickle.dump(reg, f)

    # print('Training Score:', reg.score(Xtrain, Ytrain))

    generate_csv(reg,Xtest[:, [0] + list(range(5000, Xtest.shape[1]))])
