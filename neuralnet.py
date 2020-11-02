# *- coding: utf-8 -*-
#######################################################
# Author: Jeongyun Lee
# Date: 10/30/2020
# Description: Label images of handwritten letters by implementing a Neural Network from scratch. Implemented all of the functions needed to initialize, train, evaluate, and make predictions with the network. Datasets We will be using a subset of an Optical Character Recognition (OCR) dataset. This data includes images of all 26 handwritten letters; for this assignment specifically, subset will include only the letters “a,” “e,” “g,” “i,” “l,” “n,” “o,” “r,” “t,” and “u.” 
######################################################

import numpy as np
import sys

#format data set
def splitData(filename):
    data = np.genfromtxt(filename, delimiter=',').astype(int)
    xS = data[:, 1:]
    yS = data[:, 0]
    xBias = np.ones([len(xS), 1])
    x = np.hstack((xBias, xS))
    yHot = np.eye(10)[yS]
    return yS, yHot, x

#initialize alpha: if intFlag given by command line is 1, randomly create, otherwise set it to zeros
def initAlpha(x, y):
    D = hidden
    M = x.shape[1] - 1
    if intFlag == 1:
        #initialize the first elements to 0, but random for the rest
        alphaS = np.random.uniform(-0.1, 0.1, (D, M))
        bias = np.zeros((D,1))
        alpha = np.hstack((bias, alphaS))
    else:
        #initialize to all zeros
        alpha = np.zeros((D, M + 1))
    return alpha

#initialize beta: if intFlag given by command line is 1, randomly create, otherwise set it to zeros
def initBeta(x, y):
    K = 10
    D = hidden
    if intFlag == 1:
        betaS = np.random.uniform(-0.1, 0.1, (K, D))
        bias = np.zeros((K, 1))
        beta = np.hstack((bias, betaS))
    else:
        beta = np.zeros((K, D+1))
    return beta

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x/sum(exp_x)
    return softmax_x

def J(y,yHat):
    return -np.dot(y,np.log(yHat))

#forward propagation
def NNForward(x, y, alpha, beta):
    a = np.dot(alpha, x)
    zS = sigmoid(a)
    z = np.insert(zS, 0, 1, 0)
    b = np.dot(beta, z)
    yHat = softmax(b)
    loss = J(y.T, yHat)
    obj = (a, zS, z, b, yHat, loss)
    return obj

#backward propagation
def NNBackward(x, y, alpha, beta, obj):
    a, zS, z, b, yHat, loss = obj
    gB = yHat - y
    gBeta = np.dot(z, gB.T).T
    gZ = np.dot(beta[:, 1:].T, gB)
    gA = gZ * (zS * (1 - zS))
    gAlpha = np.dot(x, gA.T).T
    return gAlpha, gBeta

#Updating alpha and beta matrices using gradient decent
def SGD(x, y, alpha, beta):
    obj = NNForward(x, y, alpha, beta)
    gAlpha, gBeta = NNBackward(x, y, alpha, beta, obj)
    alpha -= learnR * gAlpha
    beta -= learnR * gBeta
    return alpha, beta

def train(x, y, alpha, beta):
    for i in range(x.shape[0]):
        xElem = x[i, :].reshape(-1,1)
        yElem = y[i, :].reshape(-1,1)
        alpha, beta = SGD(xElem, yElem, alpha, beta)
    return alpha, beta

def predict(x, y, alpha, beta):
    result = []
    for i in range(x.shape[0]):  
        xElem = x[i, :].reshape(-1, 1)
        yElem = y[i, :].reshape(-1, 1)
        obj = NNForward(xElem, yElem, alpha, beta)
        a, zS, z, b, yHat, loss = obj
        result.append(np.argmax(yHat))
    return result

def loss(x, y, alpha, beta):
    lossResult  = []
    for i in range(x.shape[0]):  
        xElem = x[i, :].reshape(-1, 1)
        yElem = y[i, :].reshape(-1, 1)
        obj = NNForward(xElem, yElem, alpha, beta)
        a, zS, z, b, yHat, l = obj
        lossResult.append(l)
    return np.mean(lossResult)

def error(yHat, y):
    count = 0
    for i in range(len(yHat)):
        if yHat[i] != y[i]:
            count += 1
    return count / len(yHat)

def writeError(file, trainE, testE):
    with open(file, 'w') as fd:
        fd.write('error(train): %f\n' % trainE)
        fd.write('error(test): %f' % testE)

def writeLabel(file, labels):
    with open(file, 'w') as fd:
        for label in labels:
            fd.write(str(label) + '\n')

def writeCross(file, epoch, trainL, testL):
    with open(file, 'w') as fd:
        fd.write('epoch=%d crossentropy(train) : %f\n' % (epoch + 1, trainL))
        fd.write('epoch=%d crossentropy(test) : %f\n' % (epoch + 1, testL))
    
if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden = sys.argv[7]
    int_flag = sys.argv[8]
    learning_rate = sys.argv[9]
    epochNum = int(num_epoch)
    hidden = int(hidden)
    intFlag = int(int_flag)
    learnR = float(learning_rate)

    #Format the dataset and alpha/beta matrix
    train, trainY, trainX = splitData(train_input)
    test, testY, testX = splitData(test_input)
    alpha = initAlpha(trainX, trainY)
    beta = initBeta(trainX, trainY)

    #train and calculate loss
    for epoch in range(epochNum):
        #update alpha, beta
        for i in range(trainX.shape[0]):
            xElem = trainX[i, :].reshape(-1,1)
            yElem = trainY[i, :].reshape(-1,1)
            alpha, beta = SGD(xElem, yElem, alpha, beta)
        #calculate train and test loss
        trainL = loss(trainX, trainY, alpha, beta)
        testL = loss(testX, testY, alpha, beta)
        #write cross entropy for each epoch to metrics file
        writeCross(metrics, epoch, trainL, testL)
    #make predictions
    trainHat = predict(trainX, trainY, alpha, beta)
    testHat = predict(testX, testY, alpha, beta)
    #calculate and write error rates
    trainE = error(trainHat, train)
    testE = error(testHat, test)
    writeError(metrics, trainE, testE)
    #write labels 
    writeLabel(train_out, trainHat)
    writeLabel(test_out, testHat)