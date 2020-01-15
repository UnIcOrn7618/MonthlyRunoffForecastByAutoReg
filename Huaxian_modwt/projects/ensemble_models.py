import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__')) 

import sys
sys.path.append(root_path)
from tools.ensembler import ensemble
from Huaxian_modwt.projects.variables import variables

# Set the project parameters
ORIGINAL = 'HuaxianRunoff1951-2018(1953-2018).xlsx'
STATION = 'Huaxian'
DECOMPOSER = 'modwt' 
PREDICTOR = 'esvr' # esvr or gbrt or lstm
PREDICT_PATTERN = 'one_step_3_ahead_forecast_pacf' # hindcast or forecast
for PREDICT_PATTERN in [
    'one_step_1_ahead_forecast_pacf',
    'one_step_1_ahead_forecast_pacf_pca30',
    'one_step_1_ahead_forecast_pacf_pca31',
    'one_step_1_ahead_forecast_pacf_pca32',
    'one_step_1_ahead_forecast_pacf_pca33',
    'one_step_1_ahead_forecast_pacf_pca34',
    'one_step_1_ahead_forecast_pacf_pca35',
    'one_step_1_ahead_forecast_pacf_pca36',
    'one_step_1_ahead_forecast_pacf_pca37',
    'one_step_1_ahead_forecast_pacf_pca38',
    'one_step_1_ahead_forecast_pacf_pca39',
    'one_step_1_ahead_forecast_pacf_pca40',
    'one_step_1_ahead_forecast_pacf_pca41',
    'one_step_1_ahead_forecast_pacf_pca42',
    'one_step_1_ahead_forecast_pacf_pca43',
    'one_step_1_ahead_forecast_pacf_pca44',
    'one_step_1_ahead_forecast_pacf_pca45',
    'one_step_1_ahead_forecast_pacf_pca46',
    'one_step_1_ahead_forecast_pacf_pcamle',
    'one_step_3_ahead_forecast_pacf',
    'one_step_5_ahead_forecast_pacf',
    'one_step_7_ahead_forecast_pacf',
    'one_step_9_ahead_forecast_pacf',
    'one_step_3_ahead_forecast_pearson0.2',
    'one_step_5_ahead_forecast_pearson0.2',
    'one_step_7_ahead_forecast_pearson0.2',
    'one_step_9_ahead_forecast_pearson0.2',
]:
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
