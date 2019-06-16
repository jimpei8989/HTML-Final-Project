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
                    nn.LeakyReLU(0.1126),
                    )

        self.DNN = nn.Sequential(
                DenseBN(10000, 1024),
                DenseBN(1024, 512),
                nn.Dropout(0.3),
                DenseBN(512, 512),
                DenseBN(512, 256),
                nn.Dropout(0.3),
                DenseBN(256, 256),
                nn.Linear(256, 3),
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    loadTS = time.time()
    print('--- Begin Loading Data ---')
    X, Y, testX = LoadAll(dataDir)
    print('--- End Loading Data (Elapsed {:2.3f})'.format(time.time() - loadTS))

    epochs, batchSize = 1000, 256

    mean = np.mean(np.concatenate([X, testX], axis = 0), axis = 0)
    std = np.std(np.concatenate([X, testX], axis = 0), axis = 0)
    X, testX = (X - mean) / std, (testX - mean) / std

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

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

        for epoch in range(1, epochs + 1):
            beginTS = time.time()
            trainLoss, validLoss = 0, 0
            trainScore, validScore = np.zeros(shape = 3), np.zeros(shape = 3)

            model.train()
            for (ipt, opt) in trainLoader:
                optimizer.zero_grad()

                pred = model(ipt.cuda())
                loss = criterion(pred, opt.cuda())
                loss.backward()
                optimizer.step()

                trainLoss += loss.item() * ipt.shape[0] / trainNum
                trainScore += score(pred.cpu().data.numpy(), opt.data.numpy()) * ipt.shape[0] / trainNum

            model.eval()
            for (ipt, opt) in validLoader:
                pred = model(ipt.cuda())
                loss = criterion(pred, opt.cuda())

                validLoss += loss.item() * ipt.shape[0] / validNum
                validScore += score(pred.cpu().data.numpy(), opt.data.numpy()) * ipt.shape[0] / validNum

            print('Epoch: {0:3d}/{1:3d} (Elapsed {2:2.3}s)\n ~~> Train Loss: {3:3.5f} | Train Score: {4[0]:3.5f} / {4[1]:3.5f} / {4[2]:3.5f}\n ~~> Valid Loss: {5:3.5f} | Valid Score: {6[0]:3.5f} / {6[1]:3.5f} / {6[2]:3.5f}'.format(epoch, epochs, time.time() - beginTS, trainLoss, trainScore, validLoss, validScore))

            if epoch % 20 == 0:
                torch.save(model, 'models/model_{:03d}'.format(epoch))

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

