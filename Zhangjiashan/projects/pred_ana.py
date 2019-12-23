import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
# grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_path = root_path+'/Zhangjiashan/projects/lstm/'
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
        train_rmse.append(data['train_rmse'][0])
        dev_rmse.append(data['dev_rmse'][0])
        test_rmse.append(data['test_rmse'][0])
        train_r2.append(data['train_r2'][0])
        dev_r2.append(data['dev_r2'][0])
        test_r2.append(data['test_r2'][0])

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
metrics_df.to_csv(model_path+'metrics.csv')
