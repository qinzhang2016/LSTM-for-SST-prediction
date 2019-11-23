#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 08:48:31 2017

@author: crystal
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:35:15 2017
#predict 7 days values at a time, and put back one day's true value 
#and predict the next 7
@author: crystal
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy.io as sio

TIMESTEPS = 30
OUTPUTDIM = 7
NUM_HIDDENUNITS=6
NUM_EPOCH = 100
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

rMSE_total=0
acc_total=0

f=open('results_7.txt','a')
f.write('30-->7 hidden=6, l_fc=1[7], l_r=1 \n')
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
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        # create and fit the LSTM network
        
        model = Sequential()
        model.add(LSTM(NUM_HIDDENUNITS, batch_input_shape=(batch_size, look_back_x, 1), dropout_W=0, dropout_U=0, stateful=True, return_sequences=False))
        #model.add(LSTM(NUM_HIDDENUNITS, dropout_W=0, dropout_U=0, stateful=True, return_sequences=True))
        #model.add(LSTM(NUM_HIDDENUNITS, dropout_W=0, dropout_U=0,stateful=True))
        model.add(Dense(output_dim=OUTPUTDIM))
        model.add(Activation("relu"))
        model.compile(loss='mean_squared_error', optimizer='Nadam',metrics=['mse'])
        #model.compile(loss='kullback_leibler_divergence', optimizer='adam')
        hist=model.fit(trainX, trainY, nb_epoch=NUM_EPOCH, batch_size=batch_size, verbose=2, shuffle=False,validation_split=0.115)
        # make predictions0
        testPredict = model.predict(testX, batch_size=batch_size)
        print model.evaluate(testX,testY,batch_size=batch_size)
        
        #transform to original scale
        testY_=scaler.inverse_transform(testY)
        testPredict_=scaler.inverse_transform(testPredict)
        
        #plot
        filename=str(i) + "_" + str(j)
        fig, ax = plt.subplots(1)
        test_values = testY_[:,0].reshape(-1,1).flatten()
        plot_test, = ax.plot(test_values)
        predicted_values = testPredict_[:,0].reshape(-1,1).flatten()
        plot_predicted, = ax.plot(predicted_values)
        plt.title('SST Predictions')
        plt.legend([plot_predicted, plot_test],['predicted', 'true value'])
        plt.savefig(filename+'_predict')
        plt.show()
    
        
        #scores
        MSE = mean_squared_error(testPredict_, testY_)
        print ("MSE: %f" % MSE)
        rMSE = math.sqrt(MSE)
        print('rMSE:%f' % rMSE )
        pcc=np.corrcoef(testPredict_,testY_,rowvar=0)[0,1]
        print ("PCC: %f" % pcc)
        acc=1-np.mean(np.abs(testPredict_-testY_)/testY_)
        print ("ACC: %f" % acc)
        #sum
        rMSE_total=rMSE_total+rMSE
        acc_total=acc_total+acc
        
        #training epoch
        fig, ax = plt.subplots(1)
        loss,=ax.plot(hist.history["loss"])
        val_loss,=ax.plot(hist.history["val_loss"])
        plt.title('training process')
        plt.legend([loss, val_loss],['loss', 'val loss'])
        plt.savefig(filename+'_train')
        plt.show()
    
        
        #write to file
        f.write(str(MSE) + ' ' + str(rMSE) + ' ' + str(pcc) + ' ' + str(acc)  + '\n')

rMSE_ave=rMSE_total/(w*h)
acc_ave=acc_total/(w*h)
f.write('\n average rMSE   ACC \n')
f.write(str(rMSE_ave) + ' ' + str(acc_ave))
f.close()
