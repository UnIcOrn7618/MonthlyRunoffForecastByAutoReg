import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 6
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
results_path = root_path+'/results_analysis/results/'
print("root path:{}".format(root_path))
sys.path.append(root_path)
from results_reader import read_two_stage, read_pure_esvr

h_records, h_predictions, h_r2, h_nrmse, h_mae, h_mape, h_ppts, h_timecost = read_pure_esvr(
    "Huaxian")
x_records, x_predictions, x_r2, x_nrmse, x_mae, x_mape, x_ppts, x_timecost = read_pure_esvr(
    "Xianyang")
z_records, z_predictions, z_r2, z_nrmse, z_mae, z_mape, z_ppts, z_timecost = read_pure_esvr(
    "Zhangjiashan")

h_vmd_records, h_vmd_predictions, h_vmd_r2, h_vmd_nrmse, h_vmd_mae, h_vmd_mape, h_vmd_ppts, h_vmd_timecost = read_two_stage(
    station="Huaxian", decomposer="vmd", predict_pattern="one_step_1_month_forecast")
x_vmd_records, x_vmd_predictions, x_vmd_r2, x_vmd_nrmse, x_vmd_mae, x_vmd_mape, x_vmd_ppts, x_vmd_timecost = read_two_stage(
    station="Xianyang", decomposer="vmd", predict_pattern="one_step_1_month_forecast")
z_vmd_records, z_vmd_predictions, z_vmd_r2, z_vmd_nrmse, z_vmd_mae, z_vmd_mape, z_vmd_ppts, z_vmd_timecost = read_two_stage(
    station="Zhangjiashan", decomposer="vmd", predict_pattern="one_step_1_month_forecast")

h_eemd_records, h_eemd_predictions, h_eemd_r2, h_eemd_nrmse, h_eemd_mae, h_eemd_mape, h_eemd_ppts, h_eemd_timecost = read_two_stage(
    station="Huaxian", decomposer="eemd", predict_pattern="one_step_1_month_forecast")
x_eemd_records, x_eemd_predictions, x_eemd_r2, x_eemd_nrmse, x_eemd_mae, x_eemd_mape, x_eemd_ppts, x_eemd_timecost = read_two_stage(
    station="Xianyang", decomposer="eemd", predict_pattern="one_step_1_month_forecast")
z_eemd_records, z_eemd_predictions, z_eemd_r2, z_eemd_nrmse, z_eemd_mae, z_eemd_mape, z_eemd_ppts, z_eemd_timecost = read_two_stage(
    station="Zhangjiashan", decomposer="eemd", predict_pattern="one_step_1_month_forecast")

h_ssa_records, h_ssa_predictions, h_ssa_r2, h_ssa_nrmse, h_ssa_mae, h_ssa_mape, h_ssa_ppts, h_ssa_timecost = read_two_stage(
    station="Huaxian", decomposer="ssa", predict_pattern="one_step_1_month_forecast")
x_ssa_records, x_ssa_predictions, x_ssa_r2, x_ssa_nrmse, x_ssa_mae, x_ssa_mape, x_ssa_ppts, x_ssa_timecost = read_two_stage(
    station="Xianyang", decomposer="ssa", predict_pattern="one_step_1_month_forecast")
z_ssa_records, z_ssa_predictions, z_ssa_r2, z_ssa_nrmse, z_ssa_mae, z_ssa_mape, z_ssa_ppts, z_ssa_timecost = read_two_stage(
    station="Zhangjiashan", decomposer="ssa", predict_pattern="one_step_1_month_forecast")

h_dwt_records, h_dwt_predictions, h_dwt_r2, h_dwt_nrmse, h_dwt_mae, h_dwt_mape, h_dwt_ppts, h_dwt_timecost = read_two_stage(
    station="Huaxian", decomposer="dwt", predict_pattern="one_step_1_month_forecast")
x_dwt_records, x_dwt_predictions, x_dwt_r2, x_dwt_nrmse, x_dwt_mae, x_dwt_mape, x_dwt_ppts, x_dwt_timecost = read_two_stage(
    station="Xianyang", decomposer="dwt", predict_pattern="one_step_1_month_forecast")
z_dwt_records, z_dwt_predictions, z_dwt_r2, z_dwt_nrmse, z_dwt_mae, z_dwt_mape, z_dwt_ppts, z_dwt_timecost = read_two_stage(
    station="Zhangjiashan", decomposer="dwt", predict_pattern="one_step_1_month_forecast")

index = [
    "Huaxian", "Huaxian-vmd", "Huaxian-eemd", "Huaxian-ssa", "Huaxian-dwt",
    "Xianyang", "Xianyang-vmd", "Xianyang-eemd", "Xianyang-ssa", "Xianyang-dwt",
    "Zhangjiashan", "Zhangjiashan-vmd", "Zhangjiashan-eemd", "Zhangjiashan-ssa", "Zhangjiashan-dwt"]
metrics_dict = {
    "r2": [h_r2, h_vmd_r2, h_eemd_r2, h_ssa_r2, h_dwt_r2,
           x_r2, x_vmd_r2, x_eemd_r2, x_ssa_r2, x_dwt_r2,
           z_r2, z_vmd_r2, z_eemd_r2, z_ssa_r2, z_dwt_r2, ],
    "rmse": [h_nrmse, h_vmd_nrmse, h_eemd_nrmse, h_ssa_nrmse, h_dwt_nrmse,
             x_nrmse, x_vmd_nrmse, x_eemd_nrmse, x_ssa_nrmse, x_dwt_nrmse,
             z_nrmse, z_vmd_nrmse, z_eemd_nrmse, z_ssa_nrmse, z_dwt_nrmse, ],
    "mae": [h_mae, h_vmd_mae, h_eemd_mae, h_ssa_mae, h_dwt_mae,
            x_mae, x_vmd_mae, x_eemd_mae, x_ssa_mae, x_dwt_mae,
            z_mae, z_vmd_mae, z_eemd_mae, z_ssa_mae, z_dwt_mae, ],
    "mape": [h_mape, h_vmd_mape, h_eemd_mape, h_ssa_mape, h_dwt_mape,
             x_mape, x_vmd_mape, x_eemd_mape, x_ssa_mape, x_dwt_mape,
             z_mape, z_vmd_mape, z_eemd_mape, z_ssa_mape, z_dwt_mape, ],
    "ppts": [h_ppts, h_vmd_ppts, h_eemd_ppts, h_ssa_ppts, h_dwt_ppts,
             x_ppts, x_vmd_ppts, x_eemd_ppts, x_ssa_ppts, x_dwt_ppts,
             z_ppts, z_vmd_ppts, z_eemd_ppts, z_ssa_ppts, z_dwt_ppts, ],
    "time_cost": [h_timecost, h_vmd_timecost, h_eemd_timecost, h_ssa_timecost, h_dwt_timecost,
                  x_timecost, x_vmd_timecost, x_eemd_timecost, x_ssa_timecost, x_dwt_timecost,
                  z_timecost, z_vmd_timecost, z_eemd_timecost, z_ssa_timecost, z_dwt_timecost, ],
}

metrics_df = pd.DataFrame(metrics_dict, index=index)
print(metrics_df)
metrics_df.to_csv(results_path+"two_stage_decomposer_esvr_metrics.csv")

huaxian_r2 = [h_r2, h_eemd_r2, h_ssa_r2, h_ssa_r2, h_dwt_r2, ]
huaxian_nrmse = [h_nrmse, h_eemd_nrmse, h_ssa_nrmse, h_vmd_nrmse, h_dwt_nrmse, ]
huaxian_mae = [h_mae, h_eemd_mae, h_ssa_mae, h_vmd_mae, h_dwt_mae, ]
huaxian_mape = [h_mape, h_eemd_mape, h_ssa_mape, h_vmd_mape, h_dwt_mape, ]
huaxian_ppts = [h_ppts, h_eemd_ppts, h_ssa_ppts, h_vmd_ppts, h_dwt_ppts, ]
huaxian_time = [h_timecost, h_eemd_timecost,
                h_ssa_timecost, h_vmd_timecost, h_dwt_timecost, ]

xianyang_r2 = [x_r2, x_eemd_r2, x_ssa_r2, x_vmd_r2, x_dwt_r2, ]
xianyang_nrmse = [x_nrmse, x_eemd_nrmse,
                  x_ssa_nrmse, x_vmd_nrmse, x_dwt_nrmse, ]
xianyang_mae = [x_mae, x_eemd_mae, x_ssa_mae, x_vmd_mae, x_dwt_mae, ]
xianyang_mape = [x_mape, x_eemd_mape, x_ssa_mape, x_vmd_mape, x_dwt_mape, ]
xianyang_ppts = [x_ppts, x_eemd_ppts, x_ssa_ppts, x_vmd_ppts, x_dwt_ppts, ]
xianyang_time = [x_timecost, x_eemd_timecost,
                 x_ssa_timecost, x_vmd_timecost, x_dwt_timecost, ]

zhangjiashan_r2 = [z_r2, z_eemd_r2, z_ssa_r2, z_vmd_r2, z_dwt_r2, ]
zhangjiashan_nrmse = [z_nrmse, z_eemd_nrmse,
                      z_ssa_nrmse, z_vmd_nrmse, z_dwt_nrmse, ]
zhangjiashan_mae = [z_mae, z_eemd_mae, z_ssa_mae, z_vmd_mae, z_dwt_mae, ]
zhangjiashan_mape = [z_mape, z_eemd_mape, z_ssa_mape, z_vmd_mape, z_dwt_mape, ]
zhangjiashan_ppts = [z_ppts, z_eemd_ppts, z_ssa_ppts, z_vmd_ppts, z_dwt_ppts, ]
zhangjiashan_time = [z_timecost, z_eemd_timecost,
                     z_ssa_timecost, z_vmd_timecost, z_dwt_timecost, ]


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height/2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def autolabels(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height, 2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
        )


#########################################################################################
metrics_lists = [
    [huaxian_r2, xianyang_r2, zhangjiashan_r2],
    [huaxian_nrmse, xianyang_nrmse, zhangjiashan_nrmse],
    [huaxian_mae, xianyang_mae, zhangjiashan_mae],
    [huaxian_mape, xianyang_mape, zhangjiashan_mape],
    [huaxian_ppts, xianyang_ppts, zhangjiashan_ppts],
    [huaxian_time, xianyang_time, zhangjiashan_time],
]
stations = ['Huaxian', 'Xianyang', 'Zhangjiashan']
pos = [2, 4, 6, 8, 10]
print(pos)
width = 0.5
action = [-1, 0, 1]
ylims = [
    [0, 1.2],
    [0, 1.7],
    [0, 3.3],
    [0, 570],
    [0, 90],
    [0, 360],
]
labels = ['SVR', 'EEMD-SVR', 'SSA-SVR', 'VMD-SVR', 'DWT-SVR']
y_labels = [
    r"$NSE$", r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$", r"$MAE(10^8m^3)$", r"$MAPE(\%)$", r"$Time(s)$"
]
density = 5
hatch_str = ['/'*density, 'x'*density, '|'*density]
fig = plt.figure(figsize=(7.48, 7.48))
for i in range(len(metrics_lists)):
    ax = fig.add_subplot(3, 2, i+1)
    for j in range(len(metrics_lists[i])):
        bars = ax.bar([p+action[j]*width for p in pos],
                      metrics_lists[i][j], width, alpha=0.5, label=stations[j])
        for bar in bars:
            bar.set_hatch(hatch_str[j])
        # autolabels(bars,ax)
    # ax.set_ylim(ylims[i])
    ax.set_ylabel(y_labels[i])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45)
    if i == 0:
        ax.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.6, 1.25),
            # bbox_transform=plt.gcf().transFigure,
            ncol=3,
            shadow=False,
            frameon=True,
        )
plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98,
                    top=0.95, hspace=0.5, wspace=0.25)
plt.savefig(graphs_path+'two_stage_metrics.eps', format='EPS', dpi=2000)
plt.savefig(graphs_path+'two_stage_metrics.tif', format='TIFF', dpi=600)


###########################################################################################
metrics_lists = [
    [huaxian_r2, xianyang_r2, zhangjiashan_r2],
    [huaxian_nrmse, xianyang_nrmse, zhangjiashan_nrmse],
    [huaxian_ppts, xianyang_ppts, zhangjiashan_ppts],
    [huaxian_time, xianyang_time, zhangjiashan_time],
]
stations = ['Huaxian', 'Xianyang', 'Zhangjiashan']
pos = [2, 4, 6, 8, 10]
print(pos)
width = 0.5
action = [-1, 0, 1]
ylims = [
    [0, 1.2],
    [0, 3.3],
    [0, 90],
    [0, 360],
]
labels = ['SVR', 'EEMD-SVR', 'SSA-SVR', 'VMD-SVR', 'DWT-SVR']
y_labels = [
    r"$NSE$", r"$NRMSE(10^8m^3)$", r"$PPTS(5)(\%)$", r"$Time(s)$"
]
fig = plt.figure(figsize=(7.48, 5))
for i in range(len(metrics_lists)):
    ax = fig.add_subplot(2, 2, i+1)
    for j in range(len(metrics_lists[i])):
        bars = ax.bar([p+action[j]*width for p in pos],
                      metrics_lists[i][j], width, alpha=0.5, label=stations[j])
        for bar in bars:
            bar.set_hatch(hatch_str[j])
        # autolabels(bars,ax)
    # ax.set_ylim(ylims[i])
    ax.set_ylabel(y_labels[i])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45)
    if i == 0:
        ax.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.6, 1.17),
            # bbox_transform=plt.gcf().transFigure,
            ncol=3,
            shadow=False,
            frameon=True,
        )
plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98,
                    top=0.95, hspace=0.5, wspace=0.25)
plt.savefig(graphs_path+'two_stage_NSE_NRMSE_PPTS_TIMECOST.eps',
            format='EPS', dpi=2000)
plt.savefig(graphs_path+'two_stage_NSE_NRMSE_PPTS_TIMECOST.tif',
            format='TIFF', dpi=600)


###########################################################################################
nse_data = [
    [h_r2, x_r2, z_r2],
    [h_eemd_r2, x_eemd_r2, z_eemd_r2],
    [h_ssa_r2, x_ssa_r2, z_ssa_r2],
    [h_vmd_r2, x_vmd_r2, z_vmd_r2],
    [h_dwt_r2, x_dwt_r2, z_dwt_r2],
]

mean_nse = [
    sum([h_r2, x_r2, z_r2])/len([h_r2, x_r2, z_r2]),
    sum([h_eemd_r2, x_eemd_r2, z_eemd_r2]) /
    len([h_eemd_r2, x_eemd_r2, z_eemd_r2]),
    sum([h_ssa_r2, x_ssa_r2, z_ssa_r2])/len([h_ssa_r2, x_ssa_r2, z_ssa_r2]),
    sum([h_vmd_r2, x_vmd_r2, z_vmd_r2])/len([h_vmd_r2, x_vmd_r2, z_vmd_r2]),
    sum([h_dwt_r2, x_dwt_r2, z_dwt_r2])/len([h_dwt_r2, x_dwt_r2, z_dwt_r2]),
]

for i in range(1, len(mean_nse)):
    print("Compared with mean NSE of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_nse[i]-mean_nse[0])/mean_nse[0]*100))

nrmse_data = [
    [h_nrmse, x_nrmse, z_nrmse],
    [h_eemd_nrmse, x_eemd_nrmse, z_eemd_nrmse],
    [h_ssa_nrmse, x_ssa_nrmse, z_ssa_nrmse],
    [h_vmd_nrmse, x_vmd_nrmse, z_vmd_nrmse],
    [h_dwt_nrmse, x_dwt_nrmse, z_dwt_nrmse],
]

mean_nrmse = [
    sum([h_nrmse, x_nrmse, z_nrmse])/len([h_nrmse, x_nrmse, z_nrmse]),
    sum([h_eemd_nrmse, x_eemd_nrmse, z_eemd_nrmse]) /
    len([h_eemd_nrmse, x_eemd_nrmse, z_eemd_nrmse]),
    sum([h_ssa_nrmse, x_ssa_nrmse, z_ssa_nrmse]) /
    len([h_ssa_nrmse, x_ssa_nrmse, z_ssa_nrmse]),
    sum([h_vmd_nrmse, x_vmd_nrmse, z_vmd_nrmse]) /
    len([h_vmd_nrmse, x_vmd_nrmse, z_vmd_nrmse]),
    sum([h_dwt_nrmse, x_dwt_nrmse, z_dwt_nrmse]) /
    len([h_dwt_nrmse, x_dwt_nrmse, z_dwt_nrmse]),
]

for i in range(1, len(mean_nrmse)):
    print("Compared with mean NRMSE of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_nrmse[i]-mean_nrmse[0])/mean_nrmse[0]*100))

ppts_data = [
    [h_ppts, x_ppts, z_ppts],
    [h_eemd_ppts, x_eemd_ppts, z_eemd_ppts],
    [h_ssa_ppts, x_ssa_ppts, z_ssa_ppts],
    [h_vmd_ppts, x_vmd_ppts, z_vmd_ppts],
    [h_dwt_ppts, x_dwt_ppts, z_dwt_ppts],
]

mean_ppts=[
    sum([h_ppts, x_ppts, z_ppts])/len([h_ppts, x_ppts, z_ppts]),
    sum([h_eemd_ppts, x_eemd_ppts, z_eemd_ppts])/len([h_eemd_ppts, x_eemd_ppts, z_eemd_ppts]),
    sum([h_ssa_ppts, x_ssa_ppts, z_ssa_ppts])/len([h_ssa_ppts, x_ssa_ppts, z_ssa_ppts]),
    sum([h_vmd_ppts, x_vmd_ppts, z_vmd_ppts])/len([h_vmd_ppts, x_vmd_ppts, z_vmd_ppts]),
    sum([h_dwt_ppts, x_dwt_ppts, z_dwt_ppts])/len([h_dwt_ppts, x_dwt_ppts, z_dwt_ppts]),
]

for i in range(1, len(mean_ppts)):
    print("Compared with mean PPTS of SVR\nEEMD-SVR, SSA-SVR, VMD-SVR and DWT-SVR reduced by {}%".format(
        (mean_ppts[i]-mean_ppts[0])/mean_ppts[0]*100))

timecost_data = [
    [h_timecost, x_timecost, z_timecost],
    [h_eemd_timecost, x_eemd_timecost, z_eemd_timecost],
    [h_ssa_timecost, x_ssa_timecost, z_ssa_timecost],
    [h_vmd_timecost, x_vmd_timecost, z_vmd_timecost],
    [h_dwt_timecost, x_dwt_timecost, z_dwt_timecost],
]
all_datas = [
    nse_data, nrmse_data, ppts_data, timecost_data
]


x = list(range(5))
plt.figure(figsize=(3.54, 2.54))
x_s = [-0.38, 3.8, -0.38, 3.8]
y_s = [0.92, 1.28, 3, 212]
fig_ids = ['(a)', '(b)', '(c)', '(d)']
for i in range(len(all_datas)):
    ax1 = plt.subplot(2, 2, i+1)
    ax1.yaxis.grid(True)
    vplot1 = plt.violinplot(
        dataset=all_datas[i],
        positions=x,
        showmeans=True,
    )
    plt.ylabel(y_labels[i])
    if i==2 or i==3:
        plt.xticks(x, labels, rotation=45)
    else:
        plt.xticks([])
    plt.text(x_s[i], y_s[i], fig_ids[i], fontweight='normal', fontsize=7)
    for pc in vplot1['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
plt.tight_layout()
plt.savefig(graphs_path+'/two_stage_metrics_violin.eps',
            format='EPS', dpi=2000)
plt.savefig(graphs_path+'/two_stage_metrics_violin.tif',
            format='TIFF', dpi=1200)
plt.show()
