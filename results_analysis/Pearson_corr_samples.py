import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
print("root path:{}".format(root_path))
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8


vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/one_step_3_month_forecast_pre20_thresh0.2//minmax_unsample_train.csv")
dwt_train = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/one_step_3_month_forecast_pre20_thresh0.2/minmax_unsample_train.csv")


vmd_corrs = vmd_train.corr(method="pearson")
dwt_corrs = dwt_train.corr(method="pearson")


print(vmd_corrs)
plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of VMD at Huaxian station",fontsize=6)
ax=plt.imshow(vmd_corrs)
plt.xlabel(r"${P}_i$")
plt.ylabel(r"${P}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(-1,1)
plt.tight_layout()

plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of DWT at Huaxian station",fontsize=6)
ax=plt.imshow(dwt_corrs)
plt.xlabel(r"${P}_i$")
plt.ylabel(r"${P}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(-1,1)
plt.tight_layout()
plt.show()





