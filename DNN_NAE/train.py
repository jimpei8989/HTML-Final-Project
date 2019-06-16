import os, sys, time, pickle
import numpy as np
from Utils.util import *

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def DenseBN(ipt, opt, drop = 0):
            return nn.Sequential(
                    nn.Linear(ipt, opt),
                    nn.BatchNorm1d(opt),
                    nn.ReLU(),
                    nn.Dropout(drop)
                    )

        self.DNN = nn.Sequential(
                DenseBN(10000, 1024, 0.3),
                DenseBN(1024, 256, 0.4),
                DenseBN(256, 256, 0.4),
                DenseBN(256, 64, 0.4),
                DenseBN(64, 3, 0.5),
                )

    def forward(self, x):
        out = self.DNN(x)
        return out

def NAE(opt, tar):
    return torch.mean(torch.sum(torch.abs(opt - tar) / tar, dim = 1))

def score(predict, truth):
    return np.array([np.mean(np.abs(predict - truth)),
		     np.mean(np.sum(np.abs(predict - truth) @ np.array([300, 1, 200]).reshape((3, 1)), axis = 1)),
		     np.mean(np.sum(np.abs(predict - truth) / truth, axis = 1))])


def main():
    dataDir = sys.argv[1]       if len(sys.argv) != 1 else '../data/'
    modelPath = sys.argv[2]     if len(sys.argv) != 1 else 'model.pkl'
    predictPath = sys.argv[3]   if len(sys.argv) != 1 else 'predict.csv'

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    loadTS = time.time()
    print('--- Begin Loading Data ---')
    X, Y, testX = LoadAll(dataDir)
    print('--- End Loading Data (Elapsed {:2.3f})'.format(time.time() - loadTS))

    epochs, batchSize = 1000, 256

    trainX, validX, trainY, validY = train_test_split(X, Y, test_size = 0.2, random_state = lucky_num)
    trainNum, validNum = trainX.shape[0], validX.shape[0]

    del X, Y
    
    trainSet = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
    validSet = TensorDataset(torch.Tensor(validX), torch.Tensor(validY))
    testSet = TensorDataset(torch.Tensor(testX))
    trainLoader = DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 16)
    validLoader = DataLoader(validSet, batch_size = batchSize, shuffle = False, num_workers = 16)
    testLoader = DataLoader(testSet)

    del trainX, validX, testX

    model = Model().cuda()
    try:
        model = torch.load(modelPath)
        print('> Load Model from {}'.format(modelPath))
    except:
        print('Train Model')

        criterion = NAE
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

        for epoch in range(1, epochs + 1):
            beginTS = time.time()
            trainScore, validScore = np.zeros(shape = 3), np.zeros(shape = 3)

            model.train()
            for (ipt, opt) in trainLoader:
                optimizer.zero_grad()

                pred = model(ipt.cuda())
                loss = criterion(pred, opt.cuda())
                loss.backward()
                optimizer.step()

                trainScore += score(pred.cpu().data.numpy(), opt.data.numpy()) * ipt.shape[0] / trainNum

            model.eval()
            for (ipt, opt) in validLoader:
                pred = model(ipt.cuda())
                validScore += score(pred.cpu().data.numpy(), opt.data.numpy()) * ipt.shape[0] / validNum

            print('Epoch: {0:3d}/{1:3d}\n ~~>TrnLoss: {2[0]:3.5f} / {2[1]:3.5f} / {2[2]:3.5f}\t\tVldLoss: {3[0]:3.5f} / {3[1]:3.5f} / {3[2]:3.5f}'.format(epoch, epochs, trainScore, validScore))
        torch.save(model, modelPath)
            
    # Evaluation
    evalTS = time.time()
    print('--- Begin Evaluation & Prediction ---')

    trainPredict = np.concatenate([model(x.cuda()).cpu().data.numpy() for (x, y) in trainLoader], axis = 0)
    print(trainPredict.shape)
    print('> Training Score: {0[0]:3.6f}/{0[1]:3.6f}/{0[2]:3.6f}'.format(score(trainPredict, trainY)))
    validPredict = np.concatenate([model(x.cuda()).cpu().data.numpy() for (x, y) in validLoader], axis = 0)
    print('> Validation Score: {0[0]:3.6f}/{0[1]:3.6f}/{0[2]:3.6f}'.format(score(validPredict, validY)))

    testPredict = np.concatenate([model(x.cuda()).cpu().data.numpy() for (x, ) in testLoader], axis = 0)
    with open(predictPath, 'w') as f:
        f.write('\n'.join(','.join(str(e) for e in y) for y in testPredict))

    print('--- End Evaluation & Prediction (Elapsed {:2.3f}s) ---'.format(time.time() - evalTS))

if __name__ == "__main__":
    main()

