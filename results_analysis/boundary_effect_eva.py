import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'
print("root path:{}".format(root_path))
sys.path.append(root_path+'/tools/')
from results_reader import read_two_stage_traindev_test,read_pure_esvr
h_eemd = pd.read_csv(
    root_path+"/Huaxian_eemd/projects/esvr/one_step_1_month_forecast_traindev_test/Huaxian_eemd_esvr_one_step_1_month_forecast_traindev_test.csv")
h_ssa = pd.read_csv(
    root_path+"/Huaxian_ssa/projects/esvr/one_step_1_month_forecast_traindev_test/Huaxian_ssa_esvr_one_step_1_month_forecast_traindev_test.csv")
h_vmd = pd.read_csv(
    root_path+"/Huaxian_vmd/projects/esvr/one_step_1_month_forecast_traindev_test/Huaxian_vmd_esvr_one_step_1_month_forecast_traindev_test.csv")
h_wd = pd.read_csv(
    root_path+"/Huaxian_wd/projects/esvr/db10-lev2/one_step_1_month_forecast_traindev_test/Huaxian_wd_esvr_one_step_1_month_forecast_traindev_test.csv")
x_eemd = pd.read_csv(
    root_path+"/Xianyang_eemd/projects/esvr/one_step_1_month_forecast_traindev_test/Xianyang_eemd_esvr_one_step_1_month_forecast_traindev_test.csv")
x_ssa = pd.read_csv(
    root_path+"/Xianyang_ssa/projects/esvr/one_step_1_month_forecast_traindev_test/Xianyang_ssa_esvr_one_step_1_month_forecast_traindev_test.csv")
x_vmd = pd.read_csv(
    root_path+"/Xianyang_vmd/projects/esvr/one_step_1_month_forecast_traindev_test/Xianyang_vmd_esvr_one_step_1_month_forecast_traindev_test.csv")
x_wd = pd.read_csv(
    root_path+"/Xianyang_wd/projects/esvr/db10-lev2/one_step_1_month_forecast_traindev_test/Xianyang_wd_esvr_one_step_1_month_forecast_traindev_test.csv")
z_eemd = pd.read_csv(
    root_path+"/Zhangjiashan_eemd/projects/esvr/one_step_1_month_forecast_traindev_test/Zhangjiashan_eemd_esvr_one_step_1_month_forecast_traindev_test.csv")
z_ssa = pd.read_csv(
    root_path+"/Zhangjiashan_ssa/projects/esvr/one_step_1_month_forecast_traindev_test/Zhangjiashan_ssa_esvr_one_step_1_month_forecast_traindev_test.csv")
z_vmd = pd.read_csv(
    root_path+"/Zhangjiashan_vmd/projects/esvr/one_step_1_month_forecast_traindev_test/Zhangjiashan_vmd_esvr_one_step_1_month_forecast_traindev_test.csv")
z_wd = pd.read_csv(
    root_path+"/Zhangjiashan_wd/projects/esvr/db10-lev2/one_step_1_month_forecast_traindev_test/Zhangjiashan_wd_esvr_one_step_1_month_forecast_traindev_test.csv")


h_eemd_dev_y, h_eemd_dev_pred, h_eemd_test_y, h_test_pred, h_eemd_metrics = read_two_stage_traindev_test(
    station="Huaxian", decomposer="eemd", predict_pattern="one_step_1_month_forecast")
h_ssa_dev_y, h_ssa_dev_pred, h_ssa_test_y, h_test_pred, h_ssa_metrics = read_two_stage_traindev_test(
    station="Huaxian", decomposer="ssa", predict_pattern="one_step_1_month_forecast")
h_vmd_dev_y, h_vmd_dev_pred, h_vmd_test_y, h_test_pred, h_vmd_metrics = read_two_stage_traindev_test(
    station="Huaxian", decomposer="vmd", predict_pattern="one_step_1_month_forecast")
h_wd_dev_y, h_wd_dev_pred, h_wd_test_y, h_test_pred, h_wd_metrics = read_two_stage_traindev_test(
    station="Huaxian", decomposer="wd", predict_pattern="one_step_1_month_forecast")

x_eemd_dev_y, x_eemd_dev_pred, x_eemd_test_y, x_test_pred, x_eemd_metrics = read_two_stage_traindev_test(
    station="Xianyang", decomposer="eemd", predict_pattern="one_step_1_month_forecast")
x_ssa_dev_y, x_ssa_dev_pred, x_ssa_test_y, x_test_pred, x_ssa_metrics = read_two_stage_traindev_test(
    station="Xianyang", decomposer="ssa", predict_pattern="one_step_1_month_forecast")
x_vmd_dev_y, x_vmd_dev_pred, x_vmd_test_y, x_test_pred, x_vmd_metrics = read_two_stage_traindev_test(
    station="Xianyang", decomposer="vmd", predict_pattern="one_step_1_month_forecast")
x_wd_dev_y, x_wd_dev_pred, x_wd_test_y, x_test_pred, x_wd_metrics = read_two_stage_traindev_test(
    station="Xianyang", decomposer="wd", predict_pattern="one_step_1_month_forecast")

z_eemd_dev_y, z_eemd_dev_pred, z_eemd_test_y, z_test_pred, z_eemd_metrics = read_two_stage_traindev_test(
    station="Zhangjiashan", decomposer="eemd", predict_pattern="one_step_1_month_forecast")
z_ssa_dev_y, z_ssa_dev_pred, z_ssa_test_y, z_test_pred, z_ssa_metrics = read_two_stage_traindev_test(
    station="Zhangjiashan", decomposer="ssa", predict_pattern="one_step_1_month_forecast")
z_vmd_dev_y, z_vmd_dev_pred, z_vmd_test_y, z_test_pred, z_vmd_metrics = read_two_stage_traindev_test(
    station="Zhangjiashan", decomposer="vmd", predict_pattern="one_step_1_month_forecast")
z_wd_dev_y, z_wd_dev_pred, z_wd_test_y, z_test_pred, z_wd_metrics = read_two_stage_traindev_test(
    station="Zhangjiashan", decomposer="wd", predict_pattern="one_step_1_month_forecast")
all_data1 = [
    [h_eemd["dev_r2"][0], x_eemd["dev_r2"][0], z_eemd["dev_r2"][0]],
    [h_eemd["test_r2"][0], x_eemd["test_r2"][0], z_eemd["test_r2"][0]],
    [h_ssa["dev_r2"][0], x_ssa["dev_r2"][0], z_ssa["dev_r2"][0]],
    [h_ssa["test_r2"][0], x_ssa["test_r2"][0], z_ssa["test_r2"][0]],
    [h_vmd["dev_r2"][0], x_vmd["dev_r2"][0], z_vmd["dev_r2"][0]],
    [h_vmd["test_r2"][0], x_vmd["test_r2"][0], z_vmd["test_r2"][0]],
    [h_wd["dev_r2"][0], x_wd["dev_r2"][0], z_wd["dev_r2"][0]],
    [h_wd["test_r2"][0], x_wd["test_r2"][0], z_wd["test_r2"][0]],
]

all_data2 = [
    [h_eemd_metrics["dev_nse"], x_eemd_metrics["dev_nse"], z_eemd_metrics["dev_nse"]],
    [h_eemd_metrics["test_nse"], x_eemd_metrics["test_nse"], z_eemd_metrics["test_nse"]],
    [h_ssa_metrics["dev_nse"], x_ssa_metrics["dev_nse"], z_ssa_metrics["dev_nse"]],
    [h_ssa_metrics["test_nse"], x_ssa_metrics["test_nse"], z_ssa_metrics["test_nse"]],
    [h_vmd_metrics["dev_nse"], x_vmd_metrics["dev_nse"], z_vmd_metrics["dev_nse"]],
    [h_vmd_metrics["test_nse"], x_vmd_metrics["test_nse"], z_vmd_metrics["test_nse"]],
    [h_wd_metrics["dev_nse"], x_wd_metrics["dev_nse"], z_wd_metrics["dev_nse"]],
    [h_wd_metrics["test_nse"], x_wd_metrics["test_nse"], z_wd_metrics["test_nse"]],
]

labels = [
    "EEMD-SVR\n(dev)",
    "EEMD-SVR\n(test)",
    "SSA-SVR\n(dev)",
    "SSA-SVR\n(test)",
    "VMD-SVR\n(dev)",
    "VMD-SVR\n(test)",
    "DWT-SVR\n(dev)",
    "DWT-SVR\n(test)",
]

all_datas = [all_data1,all_data2]
fig_index=["(a)","(b)"]
x = list(range(8))
plt.figure(figsize=(3.54, 3.))
x_s = [-0.5,-0.5]
y_s = [0.97,0.97]
id_s=['(a)','(b)']
for i in range(len(all_datas)):
    ax1 = plt.subplot(2, 1, i+1)
    ax1.yaxis.grid(True)
    # ax1.set_ylim([0.1,1.0])
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    # plt.xlabel(fig_index[i])
    plt.text(x_s[i],y_s[i],id_s[i],fontweight='normal',fontsize=7)
    plt.ylabel(r"$NSE$")
    if i==len(all_datas)-1:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

plt.subplots_adjust(left=0.12, bottom=0.16, right=0.98,top=0.99, hspace=0.05, wspace=0.2)
plt.savefig(graphs_path+'/boundary_effect_eva_nse_violin.eps',
            format='EPS', dpi=2000)
plt.savefig(graphs_path+'/boundary_effect_eva_nse_violin.tif',
            format='TIFF', dpi=1200)


all_data1 = [
    [h_eemd["dev_nrmse"][0], x_eemd["dev_nrmse"][0], z_eemd["dev_nrmse"][0]],
    [h_eemd["test_nrmse"][0], x_eemd["test_nrmse"][0], z_eemd["test_nrmse"][0]],
    [h_ssa["dev_nrmse"][0], x_ssa["dev_nrmse"][0], z_ssa["dev_nrmse"][0]],
    [h_ssa["test_nrmse"][0], x_ssa["test_nrmse"][0], z_ssa["test_nrmse"][0]],
    [h_vmd["dev_nrmse"][0], x_vmd["dev_nrmse"][0], z_vmd["dev_nrmse"][0]],
    [h_vmd["test_nrmse"][0], x_vmd["test_nrmse"][0], z_vmd["test_nrmse"][0]],
    [h_wd["dev_nrmse"][0], x_wd["dev_nrmse"][0], z_wd["dev_nrmse"][0]],
    [h_wd["test_nrmse"][0], x_wd["test_nrmse"][0], z_wd["test_nrmse"][0]],
]

all_data2 = [
    [h_eemd_metrics["dev_nrmse"], x_eemd_metrics["dev_nrmse"], z_eemd_metrics["dev_nrmse"]],
    [h_eemd_metrics["test_nrmse"], x_eemd_metrics["test_nrmse"], z_eemd_metrics["test_nrmse"]],
    [h_ssa_metrics["dev_nrmse"], x_ssa_metrics["dev_nrmse"], z_ssa_metrics["dev_nrmse"]],
    [h_ssa_metrics["test_nrmse"], x_ssa_metrics["test_nrmse"], z_ssa_metrics["test_nrmse"]],
    [h_vmd_metrics["dev_nrmse"], x_vmd_metrics["dev_nrmse"], z_vmd_metrics["dev_nrmse"]],
    [h_vmd_metrics["test_nrmse"], x_vmd_metrics["test_nrmse"], z_vmd_metrics["test_nrmse"]],
    [h_wd_metrics["dev_nrmse"], x_wd_metrics["dev_nrmse"], z_wd_metrics["dev_nrmse"]],
    [h_wd_metrics["test_nrmse"], x_wd_metrics["test_nrmse"], z_wd_metrics["test_nrmse"]],
]


nrmse_datas = [all_data1,all_data2]
fig_index=["(a)","(b)"]
x = list(range(8))
plt.figure(figsize=(3.54, 5.54))
for i in range(len(all_datas)):
    ax1 = plt.subplot(2, 1, i+1)
    ax1.yaxis.grid(True)
    vplot1 = plt.violinplot(
        dataset=nrmse_datas[i],
        positions=x,
        showmeans=True,
    )
    plt.xlabel(fig_index[i])
    plt.ylabel(r"$NRMSE$")
    plt.xticks(x, labels, rotation=45)
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
plt.legend()
plt.tight_layout()
plt.savefig(graphs_path+'/boundary_effect_eva_nrmse_violin.eps',
            format='EPS', dpi=2000)
plt.savefig(graphs_path+'/boundary_effect_eva_nrmse_violin.tif',
            format='TIFF', dpi=1200)
plt.show()
