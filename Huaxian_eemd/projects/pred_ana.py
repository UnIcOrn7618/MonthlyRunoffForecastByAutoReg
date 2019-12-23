import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = parent_path + '\\data\\'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

MODEL_ID = 8
model_path = current_path+'\\lstm-models-history\\imf'+str(MODEL_ID)+'\\'
pred_files_list=[]
train_rmse=[]
dev_rmse=[]
test_rmse=[]
train_r2=[]
dev_r2=[]
test_r2=[]
for files in os.listdir(model_path):
    if files.find('.csv')>=0 and (files.find('HISTORY')<0 and files.find('metrics')<0):
        pred_files_list.append(files)
        print(files)
        data = pd.read_csv(model_path+files)
        train_rmse.append(data['rmse_train'][0])
        dev_rmse.append(data['rmse_dev'][0])
        test_rmse.append(data['rmse_test'][0])
        train_r2.append(data['r2_train'][0])
        dev_r2.append(data['r2_dev'][0])
        test_r2.append(data['r2_test'][0])

metrics_dict = {
    'model':pred_files_list,
    'train_rmse':train_rmse,
    'dev_rmse':dev_rmse,
    'test_rmse':test_rmse,
    'train_r2':train_r2,
    'dev_r2':dev_r2,
    'test_r2':test_r2,
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(model_path+'imf'+str(MODEL_ID)+'-metrics.csv')
