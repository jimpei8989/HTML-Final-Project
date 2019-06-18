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
    step=int(sys.argv[5]) if len(sys.argv)>=6 else 1
    
    trainX, trainY = load_data(data_dir + '/X_train.npz'), load_data(data_dir + '/Y_train.npz')
    testX = load_data(data_dir + '/X_test.npz')
    print("-> Data loaded", file=sys.stderr, flush=True)
    
    try: 
        tree = load_model(model_path)
        print("Load Model")
    except FileNotFoundError:
        trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.2)
        min_train_score, min_valid_score, min_param=1e5,1e5,0
        base=2
        for m in [12,14,15]: #range(st,ed+1,step):
            for la in range(0,4):
                print('### max_depth=',m,'lambda=',base**la,'#'*5,file=sys.stderr, flush=True)
                tree = XGBoost().fit(trainX, trainY, max_depth=m,reg_lambda=base**la, tree_method='gpu_hist', n_estimators=100, min_child_weight=0.8, validX=validX, validY=validY, eval_metric='mae', early_stopping_rounds=150, subsample=0.8, silent=0, eta=0.1)
                save_model(tree, model_path+str(m)+'la'+str(base**la))
                train_score=tree.score(trainX, trainY)
                print(' Training score:', train_score ,file=sys.stderr)
                valid_score=tree.score(validX,validY)
                print(' Validation Score:', valid_score,'\n',file=sys.stderr)
                sys.stderr.flush()
                os.fsync(2)
                del tree
                if valid_score[0] < min_valid_score[0]:
                    min_train_score, min_valid_score = train_score, valid_score
                    min_param = (m,base**la)
    
    print('\n### Training score:', min_train_score,file=sys.stderr)
    print('### Validation Score:', min_valid_score, file=sys.stderr)
    generate_csv('%s%dla%d' %(model_path,*min_param), testX,'%soutput_%d_%d.csv' % (model_path, *min_param))
    print('best tree ', min_param, file=sys.stderr)

