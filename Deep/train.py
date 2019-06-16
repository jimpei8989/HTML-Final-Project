import os, sys, time, pickle
import numpy as np
from Utils.util import *

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

def score(predict, truth):
    return (np.mean(np.abs(predict - truth)),
            np.mean(np.sum(np.abs(predict - truth) @ np.array([300, 1, 200]).reshape((3, 1)), axis = 1)),
            np.mean(np.sum(np.abs(predict - truth) / truth, axis = 1)))

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def ConvBN(ipt, opt, knl, srd):
            return nn.Sequential (
                    nn.Conv1d(ipt, opt, knl, stride = srd, padding = knl // 2),
                    nn.BatchNorm1d(opt),
                    )

        self.leftDNN = nn.Sequential(
                nn.Linear(5000, 1000),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Linear(1000, 400),
                nn.LeakyReLU(negative_slope = 0.1),
                )

        self.rightCNN = nn.Sequential(
                ConvBN( 50, 100, 5, 2),     # 100 | 50
                ConvBN(100, 150, 5, 2),     # 150 | 25
                ConvBN(150, 200, 5, 2),     # 200 | 13
                nn.MaxPool1d(2),            # 200 | 6
                )

        self.DNN = nn.Sequential(
                nn.Linear(400 + 1200, 800),
                nn.ReLU(),
                nn.Linear(400, 100),
                nn.ReLU(),
                nn.Linear(100, 3)
                )

    def forward(self, x):
        lft, rgt = x[:, :5000], x[:, 5000:].reshape((-1, 50, 100))
        lft = self.leftDNN(lft)
        rgt = self.rightCNN(rgt).view(rgt.shape[0], -1)
        tot = torch.cat([lft, rgt], dim = 1)
        out = self.DNN(tot)
        return out

def score(pred, ans):
    # pred and ans should be in shape (N, 3)
    return np.mean(np.abs(pred - ans) @ np.array([300, 1, 200]).reshape((-1, 1))), np.mean(np.abs(pred - ans) / ans)

def main():
    dataDir = sys.argv[1]       if len(sys.argv) != 1 else '../data/'
    modelPath = sys.argv[2]     if len(sys.argv) != 1 else 'model.pkl'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    loadTS = time.time()
    print('--- Begin Loading Data ---')
    X, Y, testX = LoadAll(dataDir)
    print('--- End Loading Data (Elapsed {:2.3f})'.format(time.time() - loadTS))

    model = Model().cuda()
    try:
        model = torch.load(modelPath)
        print('> Load Model from {}'.format(modelPath))

    except FileNotFoundError:
        print('Train Model')

        epochs, batchSize = 125, 128
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

        trainX, validX, trainY, validY = train_test_split(X, Y, test_size = 0.2, random_state = lucky_num)
        trainNum, validNum = trainX.shape[0], validX.shape[0]

        del X, Y
        
        trainSet = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
        validSet = TensorDataset(torch.Tensor(validX), torch.Tensor(validY))
        trainLoader = DataLoader(trainSet, batch_size = batchSize, shuffle = True, num_workers = 16)
        validLoader = DataLoader(validSet, batch_size = batchSize, shuffle = False, num_workers = 16)

        for epoch in range(1, epochs + 1):
            beginTS = time.time()
            trainLoss, validLoss = np.zeros(shape = 3), np.zeros(shape = 3)

            model.train()
            for (ipt, opt) in trainLoader:
                optimizer.zero_grad()

                pred = model(ipt.cuda())
                print(pred.shape, opt.shape)
                loss = criterion(pred, opt.cuda())
                loss.backward()
                optimizer.step()

            print('Epoch: {:3d}/{}\tTrnLoss: {2[0]:3.5f}/{2[1]:3.5f}/{2[2]:3.5f}\tVldLoss: {3[0]:3.5f}/{3[1]:3.5f}/{3[2]:3.5f}'.format(epoch, epochs, trainScore, validScore))

        torch.save(model, modelPath)
            
    # Evaluation
    evaTS = time.time()
    print('--- Begin Evaluating and Predicting ---')
    print('> Training Score:\t{0[0]:2.6f}\t{0[1]:2.6f}'.format(score(Y, )))
    print('--- End Evaluating and Predicting (Elapsed {:2.3f}s)'.format(time.time() - evaTS))


if __name__ == "__main__":
    main()
