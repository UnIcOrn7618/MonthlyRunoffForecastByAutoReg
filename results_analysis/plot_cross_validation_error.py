import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

from tools.plot_utils import plot_cv_error

plot_cv_error(
    data_path=root_path+'/Huaxian_vmd/projects/esvr/one_step_1_ahead_forecast_pacf/history/',
    labels='VMD-SVR',
)