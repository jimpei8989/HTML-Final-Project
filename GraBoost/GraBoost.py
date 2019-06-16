import os, sys, pickle,gc
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

def bruteforce(trainX, trainY, validX, validY, **kwargs):
    reg = GraBoost().fit(trainX, trainY, **kwargs)
    return reg, reg.score(trainX, trainY)[0], reg.score(validX, validY)[0]


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
        print('Load Model',file=sys.stderr)
    except FileNotFoundError:
        #pca=load_model(os.path.join(data_dir,'pca_model100'))
        #print('loaded PCA',file=sys.stderr)
        trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.2)

        filenum=np.random.randint(0,1000)
        while os.path.exists('log'+str(filenum)):
            filenum=np.random.randint(0,1000)
        f=open('log'+str(filenum),'w')
        print('open log file "log'+str(filenum)+'"',file=sys.stderr)

        param_list=[{'max_depth':m, 'tol':10.0**t} for m in range(17,18,2) for t in np.arange(-5,-8,-1)]
        reg,score=None,1e5
        cpun=9
        train_step=int(1*cpun//3)
        with Parallel(n_jobs=cpun//3, backend="threading") as parallel:
            try:
                for i in range(0,len(param_list),train_step):
                    regs=parallel(delayed(bruteforce)(trainX, trainY, validX, validY, n_estimators=500, min_impurity_decrease=0.02, n_iter_no_change=7, **param) for param in param_list[i:min(i+train_step,len(param_list))])
                    for r in regs:
                        print(r[0].regs[0], '\n',r[2],' ',r[1],sep='', end='\n###\n',file=f)
                    f.flush()
                    os.fsync(f.fileno())

                    min_reg=regs[int(np.argmin(np.array([r[2] for r in regs])))]
                    if reg is None or score>min_reg[2]:
                        reg,score = min_reg[0],min_reg[2]
                        
                    del regs,min_reg
                    gc.collect()
            except MemoryError:
                print('MemoryError!')
                for r in regs:
                    print(r[0].regs[0], '\n',r[2],' ',r[1],sep='', end='\n###\n',file=f)
                f.flush()
                os.fsync(f.fileno())
                f.close()
                exit(1)

        f.close()
        print(reg.regs[0])
        save_model(reg, model_path)

    print('Training Score:', reg.score(trainX, trainY),file=sys.stderr)
    print('Validation Score:', reg.score(validX, validY),file=sys.stderr)
    #generate_csv(reg,testX)
    #print('generated csv')

