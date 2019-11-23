#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Linear regression for SST prediction
k=30
lead time = 7,30,90,365,1095
"""

#import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import pandas as pd
import math
from sklearn.svm import SVR
import matplotlib.pyplot as plt

TIMESTEPS = 120
OUTPUTDIM = 30
batch_size = 100
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back_x, look_back_y):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back_x - look_back_y -1):
        a = dataset[i:(i+look_back_x), 0]
        dataX.append(a)
        b = dataset[(i+look_back_x):(i+look_back_x+look_back_y), 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)
    
# load the dataset
sst=sio.loadmat("../data/sst_bohai.mat")
x=sst["sst"]
w, h, n =x.shape

train_size = 11435 #int(len(dataset) * 0.67)
test_size = 1095 #len(dataset) - train_size
look_back_x = TIMESTEPS
look_back_y = OUTPUTDIM

f=open('results_lr.txt','a')
f.write('SVR:120-->30 \n')
f.write('point' + ' MSE ' +' rMSE ' + ' PCC ' +' ACC ' + '\n')
for i in range(0,w):
    for j in range(0,h):
        data_sst=x[i][j]
        f.write(str(i) + '_' + str(j) + ':')
        #dataset preprocess
        dataframe =  pd.DataFrame(data_sst)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # split into train and test sets
     
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
        trainX, trainY = create_dataset(train, look_back_x, look_back_y)
        length=len(trainX)/batch_size *batch_size
        trainX=trainX[0:length]
        trainY=trainY[0:length]
        testX, testY = create_dataset(test, look_back_x, look_back_y)
        length=len(testX)/batch_size *batch_size
        testX=testX[0:length]
        testY=testY[0:length]

        trainY1=trainY[:,1]
    
        # Create linear regression object
        #regr = linear_model.LinearRegression()
        #regr.fit(trainX, trainY1)

        #svr
        regr = SVR(kernel='rbf', C=1, gamma=1.6)
        regr.fit(trainX, trainY1)
        testPredict=[]
        for k in range(0,OUTPUTDIM):
            y1=regr.predict(testX)
            testPredict.append(y1)
            testX=np.delete(testX,0,axis=1)
            testX=np.column_stack((testX,y1))
        
        testPredict=np.transpose(np.array(testPredict))

        
        #transform to original scale
        testY_=scaler.inverse_transform(testY)
        testPredict_=scaler.inverse_transform(testPredict)  
        
        #scores
        MSE = mean_squared_error(testPredict_, testY_)
        print ("MSE: %f" % MSE)
        rMSE = math.sqrt(MSE)
        print('rMSE:%f' % rMSE )
        pcc=np.corrcoef(testPredict_,testY_,rowvar=0)[0,1]
        print ("PCC: %f" % pcc)
        acc=1-np.mean(np.abs(testPredict_-testY_)/testY_)
        print ("ACC: %f" % acc)
        
        #write to file
        f.write(str(MSE) + ' ' + str(rMSE) + ' ' + str(pcc) + ' ' + str(acc)  + '\n')

        #plot
        filename=str(i) + "_" + str(j)
        fig, ax = plt.subplots(1)
        test_values = testY_[:,0].reshape(-1,1).flatten()
        plot_test, = ax.plot(test_values)
        predicted_values = testPredict_[:,0].reshape(-1,1).flatten()
        plot_predicted, = ax.plot(predicted_values)
        plt.title('SST Predictions')
        plt.legend([plot_predicted, plot_test],['predicted', 'true value'])
        plt.savefig(filename+'_predict_lr30')
        plt.show()
    

f.close()

