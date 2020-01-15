import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
import sys
sys.path.append(root_path)
from tools.ensembler import ensemble
from Huaxian_vmd.projects.variables import variables

# Set the project parameters
ORIGINAL = 'HuaxianRunoff1951-2018(1953-2018).xlsx'
STATION = 'Huaxian'
DECOMPOSER = 'vmd' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
PREDICT_PATTERN = 'one_step_9_ahead_forecast_pacf' # hindcast or forecast

ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    decomposer=DECOMPOSER,
    lags_dict = variables['lags_dict'],
    predictor=PREDICTOR,
    predict_pattern=PREDICT_PATTERN,
    test_len=variables['test_len'],
    full_len=variables['full_len'],
)
plt.show()
