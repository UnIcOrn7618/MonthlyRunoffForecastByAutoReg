import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
from variables import multi_step_lags
import sys
sys.path.append(root_path+'/tools/')
from models import multi_step_esvr


if __name__ == '__main__':
    for i in range(1,4):
        multi_step_esvr(
            root_path=root_path,
            station='Zhangjiashan',
            decomposer='wd',
            predict_pattern='multi_step_1_month_forecast',
            lags=multi_step_lags,
            model_id=i,
            n_calls=100,
        )
    