import sys
import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(root_path)
from tools.models import one_step_esvr, one_step_esvr_multi_seed
from Huaxian_vmd.projects.variables import variables

if __name__ == '__main__':

    one_step_esvr_multi_seed(
        root_path=root_path,
        station='Huaxian',
        decomposer='vmd',
        predict_pattern='one_step_1_ahead_forecast_pacf_traindev_test',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
        n_calls=100,
    )
    for leading_time in [1,3,5,7,9]:
        one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='vmd',
            predict_pattern='one_step_'+str(leading_time)+'_ahead_forecast_pacf',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
        )

    for leading_time in [3,5,7,9]:
        one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='vmd',
            predict_pattern='one_step_'+str(leading_time)+'_ahead_forecast_pearson0.2',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
        )
    one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='vmd',
            predict_pattern='one_step_1_ahead_forecast_pacf_pcamle',#+str(i),# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
        )
    num_in_one = sum(variables['lags_dict'].values())
    for n_components in range(num_in_one-16,num_in_one+1):
        one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='vmd',
            predict_pattern='one_step_1_ahead_forecast_pacf_pca'+str(n_components),# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
        )
