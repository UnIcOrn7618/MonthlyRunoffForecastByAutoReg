import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
from variables import multi_step_lags
import sys
sys.path.append(root_path+'/tools/')
from models import multi_step_gbrt


if __name__ == '__main__':
    multi_step_gbrt(
        root_path=root_path,
        station='Huaxian',
        decomposer='vmd',
        predict_pattern='forecast',
        lags=multi_step_lags,
        model_id=1
    )
    plt.show()


    
