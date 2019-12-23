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
plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8


vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
eemd_train = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
ssa_train = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
dwt_train = pd.read_csv(root_path+"/Huaxian_wd/data/db10-lev2/WD_TRAIN.csv")
vmd_train=vmd_train.drop("ORIG",axis=1)
eemd_train=eemd_train.drop("ORIG",axis=1)
ssa_train=ssa_train.drop("ORIG",axis=1)
dwt_train=dwt_train.drop("ORIG",axis=1)
vmd_corrs = vmd_train.corr(method="pearson")
eemd_corrs = eemd_train.corr(method="pearson")
ssa_corrs = ssa_train.corr(method="pearson")
dwt_corrs = dwt_train.corr(method="pearson")


print(vmd_corrs)
plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of VMD at Huaxian station",fontsize=6)
ax=plt.imshow(vmd_corrs)
plt.xlabel(r"${S}_i$")
plt.ylabel(r"${S}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(0,1)
plt.tight_layout()
# plt.show()


plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of SSA at Huaxian station",fontsize=6)
ax=plt.imshow(ssa_corrs)
plt.xlabel(r"${S}_i$")
plt.ylabel(r"${S}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(0,1)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of EEMD at Huaxian station",fontsize=6)
ax=plt.imshow(eemd_corrs)
plt.xlabel(r"${S}_i$")
plt.ylabel(r"${S}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(0,1)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(3.54,3.54))
plt.title("Pearson-Correlation for subsignals of DWT at Huaxian station",fontsize=6)
ax=plt.imshow(dwt_corrs)
plt.xlabel(r"${S}_i$")
plt.ylabel(r"${S}_j$")
plt.colorbar(ax.colorbar, fraction=0.045)
ax.colorbar.set_label("$Corr_{i,j}$")
plt.clim(0,1)
plt.tight_layout()
# plt.show()

corrs=[eemd_corrs,ssa_corrs,vmd_corrs,dwt_corrs]
titles=["EEMD","SSA","VMD","DWT",]
plt.figure(figsize=(3.54,3.4))
for i in range(len(corrs)):
    plt.subplot(2,2,i+1)
    plt.title(titles[i],fontsize=6)
    ax1=plt.imshow(corrs[i])
    plt.xlabel(r"${S}_i$")
    plt.ylabel(r"${S}_j$")
    plt.colorbar(ax1.colorbar, fraction=0.045)
    ax1.colorbar.set_label("$Corr_{i,j}$")
    plt.clim(0,1)
plt.tight_layout()
# plt.show()
series_len=[9,12,8,3]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.54,3.3))
for (ax,i) in zip(axes.flat,list(range(len(corrs)))):
    ax.set_title(titles[i],fontsize=6)
    ax.set_xlabel(r"${S}_i$")
    ax.set_ylabel(r"${S}_j$")
    im = ax.imshow(corrs[i], cmap='viridis',vmin=0, vmax=1)
    if i==1:
        ax.set_xticks(np.arange(0, series_len[i], 2))
        ax.set_yticks(np.arange(0, series_len[i], 2))
        ax.set_xticklabels(np.arange(1, series_len[i]+1, 2))
        ax.set_yticklabels(np.arange(1, series_len[i]+1, 2))
    else:
        ax.set_xticks(np.arange(0, series_len[i], 1))
        ax.set_yticks(np.arange(0, series_len[i], 1))
        ax.set_xticklabels(np.arange(1, series_len[i]+1, 1))
        ax.set_yticklabels(np.arange(1, series_len[i]+1, 1))
fig.subplots_adjust(bottom=0.08, top=0.96, left=0.1, right=0.8,wspace=0.5, hspace=0.3)
# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
cb_ax = fig.add_axes([0.83, 0.12, 0.04, 0.805])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks(np.arange(0, 1.1, 0.5))
cbar.set_label(r"$Corr_{i,j}$")
# cbar.set_ticklabels(['low', 'medium', 'high'])
# plt.savefig(graphs_path+"Pearson_corr_huaxian.eps",format="EPS",dpi=2000)
# plt.savefig(graphs_path+"Pearson_corr_huaxian.tif",format="TIFF",dpi=1200)

plt.show()





