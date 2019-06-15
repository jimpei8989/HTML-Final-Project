import xgboost as xgb
import sys, os, pickle
from joblib import Parallel, delayed
from  Utils.util import *


class XGBoost(Model):
    @staticmethod
    def get_tree(X, Y):
        param = {
                'max-depth': 3, # the maximum depth of each tree
                'eta': 0.3, # the training step for each iteration
                'silent': 1, 
                }
        print("1")
        dtrain = xgb.DMatrix(X, label=Y)
        T = 20
        tree = xgb.train(param, dtrain, T)
        return tree

    def _fit(self, train_X, train_Y, *args):
        d = train_Y.shape[1]
        self.trees = []
        for i in range(d):
            self.trees.append(XGBoost.get_tree(train_X, train_Y[:, i]))
        return self

    def _predict(self, X, *args):
        dtrain = xgb.DMatrix(X)
        return np.concatenate([tree.predict(dtrain).reshape(-1, 1) for tree in self.trees], axis = 1)

if __name__ == "__main__":
    lucky_num = int('fcf7297f5ed12f689932bddde16eb95343daeee556b6e57523a8237aa056c345', 16) % (2 ** 32)
    
    data_dir = sys.argv[1]
    model_path= sys.argv[2]
    
    train_X, train_Y = load_data(data_dir + '/X_train.npz'), load_data(data_dir + '/Y_train.npz')
    test_X = load_data(data_dir + '/X_test.npz')
    print("-> Data loaded", file=sys.stderr)
    
    try: 
        tree = load_model(model_path)
        print("Load Model")
    except FileNotFoundError:
        tree = XGBoost().fit(train_X, train_Y)
        save_model(tree, model_path)
    
    print("Training score:", tree.score(train_X, train_Y))
    generate_csv(tree, test_X)

