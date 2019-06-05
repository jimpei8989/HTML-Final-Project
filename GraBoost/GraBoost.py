import os, sys, pickle
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from Utils.util import *

class GraBoost(Model):
    @staticmethod
    def get_reg(X, Y, **kwargs):
        return GradientBoostingRegressor(loss='lad', validation_fraction=0.2, **kwargs).fit(X, Y) 
    def _fit(self, trainX, trainY, *args, **kwargs):
        self.regs=Parallel(n_jobs=3, backend="threading")(delayed(GraBoost.get_reg)(trainX, trainY[:,i], **kwargs) for i in range(trainY.shape[1]))
        return self

    def _predict(self, X, *args):
        return np.concatenate([reg.predict(X).reshape(-1,1) for reg in self.regs], axis=1)

def bruteforce(trainX, trainY, pca, validX, validY, **kwargs):
    reg = GraBoost().fit(trainX, trainY, transform_args=pca, **kwargs)
    return reg, reg.score(validX, validY)

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
        print('loaded PCA')
        trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.2)

        regs=Parallel(n_jobs=os.cpu_count()//3, backend="threading")(delayed(bruteforce)(trainX, trainY, pca, validX, validY, n_estimators=500, min_impurity_split=10**mi, max_depth=m, n_iter_no_change=n, tol=t) for mi in range(-7,0) for m in [5,7,20,50,80,100,200] for n in [3,5,7] for t in range(-5,0))
        
        scores=np.array([reg[1][0] for reg in regs ])
        reg=regs[np.argmin(scores)][0]
        print(reg)
        save_model(reg, model_path)

    print('Training Score:', reg.score(trainX, trainY))
    print('Validation Score:', reg.score(validX, validY))
    #generate_csv(reg,testX)
    #print('generated csv')

