import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from tools.models import one_step_esvr,one_step_esvr_multi_seed


if __name__ == '__main__':
    # one_step_esvr_multi_seed(
    #     root_path=root_path,
    #     station='Huaxian',
    #     decomposer='vmd',
    #     predict_pattern='one_step_1_ahead_forecast_pacf_traindev_test',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
    #     n_calls=100,
    # )

    # one_step_esvr_multi_seed(
    #     root_path=root_path,
    #     station='Huaxian',
    #     decomposer='vmd',
    #     predict_pattern='one_step_1_ahead_forecast_pacf',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
    #     n_calls=100,
    # )
    

    for leading_time in [3,5,7,9]:
        one_step_esvr_multi_seed(
            root_path=root_path,
            station='Huaxian',
            decomposer='vmd',
            predict_pattern='one_step_'+str(leading_time)+'_ahead_forecast_pacf',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
            n_calls=100,
        )


    # for i in range(15,30):
        # one_step_esvr_multi_seed(
        #     root_path=root_path,
        #     station='Huaxian',
        #     decomposer='vmd',
        #     predict_pattern='one_step_1_month_forecast_with_pca_30',#+str(i),# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
        #     n_calls=100,
        # )
    
    
    
    