#######################################################
# Author: Jeongyun Lee
# Dat: 10/30/2020
# Description: Labeling images of handwritten letters by implementing a Neural Network from scratch. Implemented all of the functions needed to initialize, train, evaluate, and make predictions with the network.
 ######################################################

import numpy as np
import sys

def splitData(filename):
    data = np.genfromtxt(filename, delimiter=',').astype(int)
    xS = data[:, 1:]
    yS = data[:, 0]
    xBias = np.ones([len(xS), 1])
    x = np.hstack((xBias, xS))
    yHot = np.eye(10)[yS]
    return yS, yHot, x

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

def NNForward(x, y, alpha, beta):
    a = np.dot(alpha, x)
    zS = sigmoid(a)
    z = np.insert(zS, 0, 1, 0)
    b = np.dot(beta, z)
    yHat = softmax(b)
    loss = J(y.T, yHat)
    obj = (a, zS, z, b, yHat, loss)
    return obj

def NNBackward(x, y, alpha, beta, obj):
    a, zS, z, b, yHat, loss = obj
    gB = yHat - y
    gBeta = np.dot(z, gB.T).T
    gZ = np.dot(beta[:, 1:].T, gB)
    gA = gZ * (zS * (1 - zS))
    gAlpha = np.dot(x, gA.T).T
    return gAlpha, gBeta

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
        print('error(train): %f' % trainE)
        print('error(test): %f' % testE)


def writeLabel(file, labels):
    with open(file, 'w') as fd:
        for label in labels:
            fd.write(str(label) + '\n')

def writeCross(file, epoch, trainL, testL):
    with open(file, 'w') as fd:
        fd.write('epoch=%d crossentropy(train) : %f\n' % (epoch + 1, trainL))
        fd.write('epoch=%d crossentropy(test) : %f\n' % (epoch + 1, testL))
        print('epoch=%d crossentropy(train) : %f\n' % (epoch + 1, trainL))
        print('epoch=%d crossentropy(test) : %f\n' % (epoch + 1, testL))
    
    

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

    #Defining initial alpha and beta 
    #Helper function for getting the M, D, K value 
    epochNum = int(num_epoch)
    hidden = int(hidden)
    intFlag = int(int_flag)
    learnR = float(learning_rate)

    train, trainY, trainX = splitData(train_input)
    test, testY, testX = splitData(test_input)
    alpha = initAlpha(trainX, trainY)
    beta = initBeta(trainX, trainY)
    for epoch in range(epochNum):
        trainLen = trainX.shape[0]
        #update alpha, beta
        for i in range(trainLen):
            xElem = trainX[i, :].reshape(-1,1)
            yElem = trainY[i, :].reshape(-1,1)
            alpha, beta = SGD(xElem, yElem, alpha, beta)
        lossResult  = []
        #calculate train loss
        for i in range(trainLen):  
            xElem = trainX[i, :].reshape(-1, 1)
            yElem = trainY[i, :].reshape(-1, 1)
            obj = NNForward(xElem, yElem, alpha, beta)
            a, zS, z, b, yHat, l = obj
            lossResult.append(l)
        trainL = np.mean(lossResult)
        testLen = testX.shape[0]
        lossResult  = []
        #calculate test loss
        for i in range(testLen):  
            xElem = testX[i, :].reshape(-1, 1)
            yElem = testY[i, :].reshape(-1, 1)
            obj = NNForward(xElem, yElem, alpha, beta)
            a, zS, z, b, yHat, l = obj
            lossResult.append(l)
        testL = np.mean(lossResult)
        print(trainL, testL)
        writeCross(metrics, epoch, trainL, testL)
    #make predictions
    trainHat = predict(trainX, trainY, alpha, beta)
    testHat = predict(testX, testY, alpha, beta)
    #calculate and write error rates
    trainE = error(trainHat, train)
    testE = error(testHat, test)
    writeError(metrics, trainE, testE)
    
    writeLabel(train_out, trainHat)
    writeLabel(test_out, testHat)