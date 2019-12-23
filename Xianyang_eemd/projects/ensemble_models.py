import matplotlib.pyplot as plt

from variables import multi_step_lags,test_len,full_len
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 

import sys
sys.path.append(root_path+'/tools/')
from ensembler import ensemble

# Set the project parameters
ORIGINAL = 'XianyangRunoff1951-2018(1953-2018).xlsx'
STATION = 'Xianyang'
DECOMPOSER = 'eemd' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
PREDICT_PATTERN = 'multi_step_1_month_forecast' # hindcast or forecast

ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    multi_step_lags=multi_step_lags,
    predictor=PREDICTOR,
    predict_pattern=PREDICT_PATTERN,
    test_len=test_len,
    full_len=full_len,
)
plt.show()
