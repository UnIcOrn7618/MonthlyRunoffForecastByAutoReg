# from sklearn.svm import SVR
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7.48,3))
# ax1 = plt.subplot2grid((1,5), (0,0), colspan=3)
# ax2 = plt.subplot2grid((1,5), (0,3),colspan=2,aspect='equal')
# plt.tight_layout()
# plt.show()


import os
root_path = os.path.dirname(os.path.abspath('__file__'))
print(os.getcwd())

import pandas as pd
data1 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_3_month_forecast_new/minmax_unsample_dev.csv')
data2 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_3_ahead_forecast_pacf/minmax_unsample_dev.csv')
print((data1-data2).sum(axis=0))

data1 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_3_month_forecast_0.2/minmax_unsample_dev.csv')
data2 = pd.read_csv(root_path+'/Huaxian_eemd/data/one_step_3_ahead_forecast_peason0.2/minmax_unsample_dev.csv')
print((data1-data2).sum(axis=0))