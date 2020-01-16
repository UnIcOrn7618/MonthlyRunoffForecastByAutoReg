import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

from tools.plot_utils import plot_cv_error

plot_cv_error(
    data_path=[
        root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/history/',
        root_path+'/Huaxian_modwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/history/',
        root_path+'/Huaxian_eemd/projects/esvr/one_step_1_ahead_forecast_pacf/history/',
        root_path+'/Huaxian_ssa/projects/esvr/one_step_1_ahead_forecast_pacf/history/',
        root_path+'/Huaxian_dwt/projects/esvr/db10-2/one_step_1_ahead_forecast_pacf/history/'

    ],
    labels=[
        'VMD-SVR of Huaxian',
        'MODWT-SVR of Huaxian',
        'EEMD-SVR of Huaxian',
        'SSA-SVR of Huaxian',
        'DWT-SVR of Huaxian',
    ],
)