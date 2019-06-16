import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from  Utils.util import *


class XGBoost(Model):
    @staticmethod
    def get_tree(X, Y, evals=(), early_stopping_rounds=None, xgb_model=None, **kwargs):
        dtrain = xgb.DMatrix(X, label=Y, nthread=-1)
        tree = xgb.train(kwargs, dtrain, num_boost_round=512, evals=evals, early_stopping_rounds=early_stopping_rounds, xgb_model=xgb_model)
        return tree

    def _fit(self, trainX, trainY, *args, validX=None, validY=None, **kwargs):
        evals=()
        if validX is not None and validY is not None:
            valid=xgb.DMatrix(validX, label=validY, nthread=-1)
            evals=[(valid, 'eval')]
        self.regs=[ XGBoost.get_tree(trainX, trainY[:,i].copy(), evals=evals, **kwargs) for i in range(trainY.shape[1])]
        return self

    def _predict(self, X, *args):
        dtrain = xgb.DMatrix(X,nthread=-1)
        return np.concatenate([tree.predict(dtrain).reshape(-1, 1) for tree in self.regs], axis = 1)

if __name__ == "__main__":
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)
    
    data_dir = sys.argv[1]
    model_path= sys.argv[2]
    st,ed = list(map(int,sys.argv[3:5]))
    
    trainX, trainY = load_data(data_dir + '/X_train.npz'), load_data(data_dir + '/Y_train.npz')
    testX = load_data(data_dir + '/X_test.npz')
    print("-> Data loaded", file=sys.stderr)
    
    try: 
        tree = load_model(model_path)
        print("Load Model")
    except FileNotFoundError:
        trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.2)
        min_tree, min_score, min_param=None,1e5,0
        base=2
        for m in range(st,ed+1,2):
<<<<<<< HEAD
            for la in range(1,5):
                print('##### max_depth=',m,'lambda=',base**la,'#'*5)
                tree = XGBoost().fit(trainX, trainY, max_depth=m,reg_lambda=base**la)#, validX=validX, validY=validY, eval_metric='mae', early_stopping_rounds=10)
=======
            for la in range(4,8):
                print('##### max_depth=',m,'lambda=',base**la,'#'*5)
                tree = XGBoost().fit(trainX, trainY, validX=validX, validY=validY, max_depth=m,reg_lambda=base**la,eval_metric='mae', early_stopping_rounds=10)
>>>>>>> 73dfc5a98cdd7b0576a6b3e28f1b6df4ee2fd178
                save_model(tree, model_path+str(m)+'la'+str(base**la))
                print('#'*5+" Training score:", tree.score(trainX, trainY),file=sys.stderr)
                score=tree.score(validX,validY)
                print('#'*5+' Validation Score:', score,file=sys.stderr)
                if score[0] < min_score:
                    min_tree = tree
                    min_score = score[0]
                    min_param = (m,base**la)
    
    print("Training score:", min_tree.score(trainX, trainY),file=sys.stderr)
    print('Validation Score:', min_tree.score(validX, validY),file=sys.stderr)
    generate_csv(min_tree, testX,'./output'+str(min_param).replace(' ','')+'.csv')
    print('best tree ', min_param)

