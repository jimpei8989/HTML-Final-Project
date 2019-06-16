import os, sys, time, pickle
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from Utils.util import *

def score(predict, truth):
    return (np.mean(np.abs(predict - truth)),
            np.mean(np.abs(predict - truth) @ np.array([300, 1, 200]).reshape((3, 1))),
            np.mean(np.abs(predict - truth) / truth))

if __name__ == "__main__":
    dataDir = sys.argv[1] if len(sys.argv) != 1 else '../data/'
    modelPath = sys.argv[2] if len(sys.argv) != 1 else 'model.pkl'
    predictPath = sys.argv[3] if len(sys.argv) != 1 else 'predict.csv'

    loadTS = time.time()
    print('--- Begin Loading Data & Creating Dataset ---')
    Xtrain, Ytrain, Xtest = LoadAll(dataDir)
    trainX, validX, trainY, validY = train_test_split(Xtrain, Ytrain, test_size = 0.2, random_state = lucky_num)
    print('--- End Loading Data & Creating Dataset (Elapsed {:2.3f}s) ---'.format(time.time() - loadTS))

    targets = ['penetration', 'mesh', 'alpha']
    weights = [300, 1, 200]

    try:
        with open(modelPath, 'rb') as f:
            model = pickle.load(f)
        print('> Load Model')

    except FileNotFoundError:
        trainTS = time.time()
        print('--- Begin Training ---')
        model = [None] * 3
        for i, name in enumerate(targets):
            model[i] = lgb.LGBMRegressor(boosting_type = 'gbdt',
                                         objective = 'regression_l1',
                                         n_estimators = 512,
                                         max_depth = 17,
                                         num_leaves = 127,
                                         subsample = 0.8,
                                         n_jobs = 16,
                                         random_state = lucky_num,
                                         silent = False
                                         )

            model[i].fit(trainX, trainY[:, i])
            predict = model[i].predict(validX).reshape((-1, 1))
            print('~> Target {:10s}: L1 = {:2.3f}'.format(name, np.mean(np.abs(predict - validY[:, i]).reshape((-1, 1)))))

        with open(modelPath, 'wb') as f:
            pickle.dump(model, f)
            
        print('--- End Training (Elapsed {:2.3f}) ---'.format(time.time() - trainTS))

    evalTS = time.time()
    print('--- Begin Evaluation & Prediction ---')

    trainPredict = np.concatenate([model[i].predict(trainX).reshape((-1, 1)) for i in range(3)], axis = 1)
    print('> Training Score: {0[0]:3.6f}/{0[1]:3.6f}/{0[2]:3.6f}'.format(score(trainPredict, trainY)))
    
    validPredict = np.concatenate([model[i].predict(validX).reshape((-1, 1)) for i in range(3)], axis = 1)
    print('> Validation Score: {0[0]:3.6f}/{0[1]:3.6f}/{0[2]:3.6f}'.format(score(validPredict, validY)))

    testPredict = np.concatenate([model[i].predict(Xtest).reshape((-1, 1)) for i in range(3)], axis = 1)

    with open(predictPath, 'w') as f:
        f.write('\n'.join(','.join(str(e) for e in y) for y in testPredict))

    print('--- End Evaluation & Prediction (Elapsed {:2.3f}s) ---'.format(time.time() - evalTS))

