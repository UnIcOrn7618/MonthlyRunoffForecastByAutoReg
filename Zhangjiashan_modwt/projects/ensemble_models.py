import matplotlib.pyplot as plt

from variables import multi_step_lags,test_len,full_len
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 

import sys
sys.path.append(root_path)
from ensembler import ensemble

# Set the project parameters
ORIGINAL = 'HuaxianRunoff1951-2018(1953-2018).xlsx'
STATION = 'Huaxian'
DECOMPOSER = 'dwt' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
PREDICT_PATTERN = 'multi_step_1_month_forecast' # hindcast or forecast

ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    multi_step_llags_dict = variables['lags_dict'],
    predictor=PREDICTOR,
    predict_pattern=PREDICT_PATTERN,
    test_len=test_len,
    full_len=full_len,
)
plt.show()
