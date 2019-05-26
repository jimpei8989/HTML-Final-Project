import os, sys, pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Utils.util import *

if __name__ == "__main__":
    # The lucky number is the sha256sum of our team name!
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)

    dataDir = sys.argv[1]
    modelPath = sys.argv[2]
    predictPath = sys.argv[3]

    Xtrain, Ytrain, Xtest = LoadAll(dataDir)
    print('-> Data Loaded', file = sys.stderr)

    try:
        with open(modelPath, 'rb') as f:
            reg = pickle.load(f)

    except FileNotFoundError:
        reg = RandomForestRegressor(n_estimators = 128, n_jobs = 20, random_state = lucky_num, verbose = 1)
        reg.fit(Xtrain, Ytrain)
        with open(modelPath, 'wb') as f:
            pickle.dump(reg, f)

    print('Training Score:', reg.score(Xtrain, Ytrain))
    YPredict = reg.predict(Xtest)

    with open(predictPath, 'w') as f:
        f.write('\n'.join(','.join(str(e) for e in y) for y in YPredict))
