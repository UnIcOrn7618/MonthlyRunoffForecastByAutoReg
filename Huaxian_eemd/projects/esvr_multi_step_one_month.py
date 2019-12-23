import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
print("root_path:{}".format(root_path))
from variables import multi_step_lags
import sys
sys.path.append(root_path+'/tools/')
from models import multi_step_esvr


if __name__ == '__main__':
    for i in range(1,10):
        multi_step_esvr(
            root_path=root_path,
            station='Huaxian',
            decomposer='eemd',
            predict_pattern='multi_step_1_month_forecast',
            lags=multi_step_lags,
            model_id=i,#1:9
            n_calls=100,
        )