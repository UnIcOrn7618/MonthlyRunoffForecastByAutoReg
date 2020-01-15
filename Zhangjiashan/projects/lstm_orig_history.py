import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import seaborn as sns
import shutil
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

root_path = os.path.dirname(os.path.abspath('__file__'))
data_path = root_path + '/Zhangjiashan/data/'
model_path = root_path+'/Zhangjiashan/projects/lstm/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
import sys
sys.path.append(root_path)
from dump_data import dum_pred_results
from plot_utils import plot_rela_pred,plot_history,plot_error_distribution,plot_convergence_,plot_evaluations_,plot_objective_
from variables import lags

"""NOTE:WILL DEPRICATED IN THE FUTURE PROJECTS"""
RE_TRAIN = False
WARM_UP = False
EARLY_STOPING = True
INITIAL_EPOCH = 6000

# For initialize weights and bias
SEED=1
# set hyper-parameters
EPS=5000    #epochs number
#########--1--###########
LR=0.007    #learnin rate 0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007,0.01, 0.03 0.1
#########--2--############
HU1 = 16    #hidden units for hidden layer 1
BS = 512    #batch size
#########--3--###########
HL = 1      #hidden layers
HU2 = 16    #hidden units for hidden layer 2
DC=0.000    #decay rate of learning rate
#########--4--###########
DR1=0.0     #dropout rate for hidden layer 1
DR2=0.0     #dropout rate for hidden layer 2

# 1.Import the sampled normalized data set from disk
train = pd.read_csv(data_path+'minmax_unsample_train.csv')
dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
test = pd.read_csv(data_path+'minmax_unsample_test.csv')

# Split features from labels
train_x = train
train_y = train.pop('Y')
train_y = train_y.as_matrix()
dev_x = dev
dev_y = dev.pop('Y')
dev_y = dev_y.as_matrix()
test_x = test
test_y = test.pop('Y')
test_y = test_y.as_matrix()
# reshape the input features for LSTM
train_x = (train_x.values).reshape(train_x.shape[0],1,train_x.shape[1])
dev_x = (dev_x.values).reshape(dev_x.shape[0],1,dev_x.shape[1])
test_x = (test_x.values).reshape(test_x.shape[0],1,test_x.shape[1])
# 2.Build LSTM model with keras
# set the hyper-parameters
LEARNING_RATE=LR
EPOCHS = EPS
BATCH_SIZE = BS
if HL==2:
    HIDDEN_UNITS = [HU1,HU2]
    DROP_RATE = [DR1,DR2]
else:
    HIDDEN_UNITS = [HU1]
    DROP_RATE = [DR1]
DECAY_RATE = DC
MODEL_NAME = 'LSTM-LR['+str(LEARNING_RATE)+\
    ']-HU'+str(HIDDEN_UNITS)+\
    '-EPS['+str(EPOCHS)+\
    ']-BS['+str(BATCH_SIZE)+\
    ']-DR'+str(DROP_RATE)+\
    '-DC['+str(DECAY_RATE)+\
    ']-SEED['+str(SEED)+']'
# RESUME_TRAINING = True
def build_model():
    if HL==2:
        model = keras.Sequential(
        [
            layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
            layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
            layers.LSTM(HIDDEN_UNITS[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
            layers.Dropout(DROP_RATE[1], noise_shape=None, seed=None),
            # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
            layers.Dense(1)
        ]
    )
    else:
        model = keras.Sequential(
            [
                layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                # layers.LSTM(HIDDEN_UNITS1,activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])), # first hidden layer if hasnext hidden layer
                # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                layers.Dense(1)
            ]
        )
    optimizer = keras.optimizers.Adam(LEARNING_RATE,
    decay=DECAY_RATE
    )
    model.compile(loss='mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_absolute_error','mean_squared_error'])
    return model
# set model's parameters restore path
cp_path = model_path+MODEL_NAME+'\\'
if not os.path.exists(cp_path):
    os.makedirs(cp_path)
checkpoint_path = model_path+MODEL_NAME+'\\cp.ckpt' #restore only the latest checkpoint after every update
# checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
checkpoint_dir = os.path.dirname(checkpoint_path)
print('checkpoint dir:{}'.format(checkpoint_dir))
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,mode='min',save_weights_only=True,verbose=1)
# cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,period=5,verbose=1)
# if not RESUME_TRAINING:
#     print("Removing previous artifacts...")
#     shutil.rmtree(checkpoint_dir, ignore_errors=True)
# else:
#     print("Resuming training...")
# initialize a new model
model = build_model()
model.summary() #print a simple description for the model
"""
# Evaluate before training or load trained weights and biases
loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
# Try the model with initial weights and biases
example_batch = train_x[:10]
example_result = model.predict(example_batch)
print(example_result)
"""
# 3.Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
files = os.listdir(checkpoint_dir)

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',min_lr=0.00001,factor=0.2, verbose=1,patience=10, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=100,restore_best_weights=True)


warm_dir = 'LSTM-LR['+str(LEARNING_RATE)+\
    ']-HU'+str(HIDDEN_UNITS)+\
    '-EPS['+str(INITIAL_EPOCH)+\
    ']-BS['+str(BATCH_SIZE)+\
    ']-DR'+str(DROP_RATE)+\
    '-DC['+str(DECAY_RATE)+\
    ']-SEED['+str(SEED)+']'
print("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
# Training models
if  RE_TRAIN: # Retraining the LSTM model
    print('retrain the model')
    if EARLY_STOPING:
        start = time.process_time()
        history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
        callbacks=[
            cp_callback,
            early_stopping,
        ])
        end = time.process_time()
        time_cost = end-start
    else:
        start = time.process_time()
        history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[cp_callback])
        end =time.process_time()
        time_cost = end-start
    # # Visualize the model's training progress using the stats stored in the history object
    hist = pd.DataFrame(history.history)
    hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
    hist['epoch']=history.epoch
    # print(hist.tail())
    plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
elif len(files)==0: # The current model has not been trained
    if os.path.exists(model_path+warm_dir) and WARM_UP: # Training the model using the trained weights and biases as initialized parameters
        print('WARM UP FROM EPOCH '+str(INITIAL_EPOCH)) # Warm up from the last epoch of the target model
        prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
        warm_path=model_path+warm_dir+'\\cp.ckpt'
        model.load_weights(warm_path)
        if EARLY_STOPING:
            start=time.process_time()
            history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
            callbacks=[
                cp_callback,
                early_stopping,
                ])
            end = time.process_time()
            time_cost = end - start + prev_time_cost
        else:
            start = time.process_time()
            history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
            callbacks=[
                cp_callback,
                ])
            end = time.process_time()
            time_cost = end - start + prev_time_cost
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    else: # Training entirely new model
        print('new train')
        if EARLY_STOPING:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[
                cp_callback,
                early_stopping,
                ])
            end = time.process_time()
            time_cost = end -start
        else:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
            callbacks=[
                cp_callback,
                ])
            end = time.process_time()
            time_cost = end - start
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
else:
    print('#'*10+'Already Trained')
    time_cost = (pd.read_csv(model_path+MODEL_NAME+'.csv')['time_cost'])[0]
    model.load_weights(checkpoint_path)
    # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
"""
# Evaluate after training or load trained weights and biases
loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
"""
# 4. Predict the model
# load the unsample data
train_predictions = model.predict(train_x).flatten()
dev_predictions = model.predict(dev_x).flatten()
test_predictions = model.predict(test_x).flatten()
# renormized the predictions and labels
# load the normalized traindev indicators
norm = pd.read_csv(data_path+'norm_unsample_id.csv')
sMax = norm['series_max'][norm.shape[0]-1]
sMin = norm['series_min'][norm.shape[0]-1]
print('Series min:{}'.format(sMin))
print('Series max:{}'.format(sMax))

train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
train_predictions = np.multiply(train_predictions + 1,sMax - sMin) / 2 + sMin
train_predictions[train_predictions<0.0]=0.0
dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
dev_predictions = np.multiply(dev_predictions + 1,sMax - sMin) / 2 + sMin
dev_predictions[dev_predictions<0.0]=0.0
test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
test_predictions = np.multiply(test_predictions + 1,sMax - sMin) / 2 + sMin
test_predictions[test_predictions<0.0]=0.0

dum_pred_results(
    path = model_path+MODEL_NAME+'.csv',
    train_y = train_y,
    train_predictions=train_predictions,
    dev_y = dev_y,
    dev_predictions = dev_predictions,
    test_y = test_y,
    test_predictions = test_predictions,
    time_cost=time_cost,
    )

plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '-TRAIN-PRED.png')
plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "-DEV-PRED.png")
plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "-TEST-PRED.png")
plot_error_distribution(test_predictions,test_y,model_path+MODEL_NAME+'-ERROR-DSTRI.png')