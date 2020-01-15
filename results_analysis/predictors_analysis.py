import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8

import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
print("root path:{}".format(root_path))
import sys
sys.path.append(root_path)
from results_reader import read_samples_num

num_samples_eemd_hua = read_samples_num(station="Huaxian",decomposer="eemd",)
num_samples_ssa_hua = read_samples_num(station="Huaxian",decomposer="ssa",)
num_samples_vmd_hua = read_samples_num(station="Huaxian",decomposer="vmd",)
num_samples_dwt_hua = read_samples_num(station="Huaxian",decomposer="dwt",)

num_samples_eemd_xian = read_samples_num(station="Xianyang",decomposer="eemd",)
num_samples_ssa_xian = read_samples_num(station="Xianyang",decomposer="ssa",)
num_samples_vmd_xian = read_samples_num(station="Xianyang",decomposer="vmd",)
num_samples_dwt_xian = read_samples_num(station="Xianyang",decomposer="dwt",)

num_samples_eemd_zhang = read_samples_num(station="Zhangjiashan",decomposer="eemd",)
num_samples_ssa_zhang = read_samples_num(station="Zhangjiashan",decomposer="ssa",)
num_samples_vmd_zhang = read_samples_num(station="Zhangjiashan",decomposer="vmd",)
num_samples_dwt_zhang = read_samples_num(station="Zhangjiashan",decomposer="dwt",)


one_month_hua_vmd = pd.read_csv(root_path+"/Huaxian_vmd/data/one_step_1_month_forecast/minmax_unsample_train.csv")
one_month_hua_vmd = one_month_hua_vmd.drop("Y",axis=1)
num_1_hua_vmd = one_month_hua_vmd.shape[1]

one_month_hua_eemd = pd.read_csv(root_path+"/Huaxian_eemd/data/one_step_1_month_forecast/minmax_unsample_train.csv")
one_month_hua_eemd = one_month_hua_eemd.drop("Y",axis=1)
num_1_hua_eemd = one_month_hua_eemd.shape[1]

one_month_hua_ssa = pd.read_csv(root_path+"/Huaxian_ssa/data/one_step_1_month_forecast/minmax_unsample_train.csv")
one_month_hua_ssa = one_month_hua_ssa.drop("Y",axis=1)
num_1_hua_ssa = one_month_hua_ssa.shape[1]

one_month_hua_dwt = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/one_step_1_month_forecast/minmax_unsample_train.csv")
one_month_hua_dwt = one_month_hua_dwt.drop("Y",axis=1)
num_1_hua_dwt = one_month_hua_dwt.shape[1]

num_1=[
    num_1_hua_eemd,
    num_1_hua_ssa,
    num_1_hua_vmd,
    num_1_hua_dwt,
]

corrs=[
    num_samples_eemd_hua,
    num_samples_ssa_hua,
    num_samples_vmd_hua,
    num_samples_dwt_hua,
    # num_samples_eemd_xian,
    # num_samples_ssa_xian,
    # num_samples_vmd_xian,
    # num_samples_dwt_xian,
    # num_samples_eemd_zhang,
    # num_samples_ssa_zhang,
    # num_samples_vmd_zhang,
    # num_samples_dwt_zhang,
]

lablels=[
    "3-month ahead",
    "5-month ahead",
    "7-month ahead",
    "9-month ahead",
]

titles=[
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]
x=[0.46,0.46,0.46,0.46]
y=[128,86,84,63]
plt.figure(figsize=(3.54, 3.0))
t= [0.1,0.2,0.3,0.4,0.5]
plt.xticks(np.arange(start=0.1,stop=0.6,step=0.1))
for i in range(len(corrs)):
    plt.subplot(len(corrs)/2,2,i+1)
    if i==3:
        plt.ylim([0,70])
    plt.text(x[i],y[i],titles[i],fontsize=7)
    if i in range(0,2):
        plt.xticks([])
    if i in range(2,4):
        plt.xlabel("Threshold")
        plt.xticks([0.1,0.2,0.3,0.4,0.5])
    if i==0 or i==2:
        plt.ylabel("Number of predictors")
    plt.axhline(num_1[i],label='1-month ahead',color='black',linestyle='--')
    for j in range(len(corrs[i])):
        plt.plot(t,corrs[i][j],label=lablels[j])
    if i==0:
        plt.legend(
            loc='upper left',
            # loc=0,
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(-0.03,1.32),
            # bbox_transform=plt.gcf().transFigure,
            ncol=3,
            shadow=False,
            frameon=True,)
plt.subplots_adjust(left=0.125, bottom=0.12, right=0.98,top=0.89, hspace=0.02, wspace=0.2)
plt.savefig(graphs_path+"predictors_num_vs_threshold_huaxian.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"predictors_num_vs_threshold_huaxian.tif",format="TIFF",dpi=1200)
plt.show()
    
