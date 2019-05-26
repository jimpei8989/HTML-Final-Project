import os, sys, pickle
import numpy as np
from sklearn.linear_model import LinearRegression

from Utils.util import *

if __name__ == "__main__":
    # The lucky number is the sha256sum of our team name!
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

    dataDir = sys.argv[1]
    modelPath = sys.argv[2]

    Xtrain, Ytrain, Xtest = LoadAll(dataDir)
    print('-> Data Loaded', file=sys.stderr)
    Xtrain=Xtrain[:, 5000:]

    reg = LinearRegression(normalize = True, n_jobs = 12)
    reg.fit(Xtrain, Ytrain)
    print('Training Score:', reg.score(Xtrain, Ytrain))

    with open(modelPath, 'wb') as f:
        pickle.dump(reg, f)
