import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
print("root_path:{}".format(root_path))
import sys
sys.path.append(root_path+'/tools/')
from models import one_step_esvr,one_step_esvr_multi_seed


if __name__ == '__main__':

    one_step_esvr(
        root_path=root_path,
        station='Huaxian',
        decomposer='eemd',
        predict_pattern='one_step_1_month_forecast_traindev_test',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
        n_calls=100,
    )

    

    # for leading_time in [3,5,7,9]:
    #     one_step_esvr_multi_seed(
    #         root_path=root_path,
    #         station='Huaxian',
    #         decomposer='eemd',
    #         predict_pattern='one_step_'+str(leading_time)+'_month_forecast_0.3',# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
    #         n_calls=100,
    #     )

    # for i in range(25,51):
        # one_step_esvr_multi_seed(
        #     root_path=root_path,
        #     station='Huaxian',
        #     decomposer='eemd',
        #     predict_pattern='one_step_1_month_forecast',#+str(n_components),# hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
        #     n_calls=100,
        # )