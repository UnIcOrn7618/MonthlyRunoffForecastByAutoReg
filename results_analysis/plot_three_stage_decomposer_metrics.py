import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
import sys
sys.path.append(root_path+'/tools/')
from results_reader import read_two_stage,read_pure_esvr

h_records,h_predictions,h_r2,h_nrmse,h_mae,h_mape,h_ppts,h_timecost=read_pure_esvr("Huaxian")
x_records,x_predictions,x_r2,x_nrmse,x_mae,x_mape,x_ppts,x_timecost=read_pure_esvr("Xianyang")
z_records,z_predictions,z_r2,z_nrmse,z_mae,z_mape,z_ppts,z_timecost=read_pure_esvr("Zhangjiashan")

huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_eemd_sum_model_test_metrics.csv')
huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_ssa_sum_model_test_metrics.csv')
huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_vmd_sum_model_test_metrics.csv')
huaxian_wd = pd.read_csv(root_path+'/Huaxian_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Huaxian_wd_sum_model_test_metrics.csv')

xianyang_eemd = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_eemd_sum_model_test_metrics.csv')
xianyang_ssa = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_ssa_sum_model_test_metrics.csv')
xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_vmd_sum_model_test_metrics.csv')
xianyang_wd = pd.read_csv(root_path+'/Xianyang_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Xianyang_wd_sum_model_test_metrics.csv')

zhangjiashan_eemd = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_eemd_sum_model_test_metrics.csv')
zhangjiashan_ssa = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_ssa_sum_model_test_metrics.csv')
zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_vmd_sum_model_test_metrics.csv')
zhangjiashan_wd = pd.read_csv(root_path+'/Zhangjiashan_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Zhangjiashan_wd_sum_model_test_metrics.csv')

huaxian_r2 = [h_r2,huaxian_eemd['test_r2'][0],huaxian_ssa['test_r2'][0],huaxian_vmd['test_r2'][0],huaxian_wd['test_r2'][0],]
huaxian_nrmse = [h_nrmse,huaxian_eemd['test_nrmse'][0],huaxian_ssa['test_nrmse'][0],huaxian_vmd['test_nrmse'][0],huaxian_wd['test_nrmse'][0],]
huaxian_mae = [h_mae,huaxian_eemd['test_mae'][0],huaxian_ssa['test_mae'][0],huaxian_vmd['test_mae'][0],huaxian_wd['test_mae'][0],]
huaxian_mape = [h_mape,huaxian_eemd['test_mape'][0],huaxian_ssa['test_mape'][0],huaxian_vmd['test_mape'][0],huaxian_wd['test_mape'][0],]
huaxian_ppts = [h_ppts,huaxian_eemd['test_ppts'][0],huaxian_ssa['test_ppts'][0],huaxian_vmd['test_ppts'][0],huaxian_wd['test_ppts'][0],]
huaxian_time = [h_timecost,huaxian_eemd['time_cost'][0],huaxian_ssa['time_cost'][0],huaxian_vmd['time_cost'][0],huaxian_wd['time_cost'][0],]

xianyang_r2 = [x_r2,xianyang_eemd['test_r2'][0],xianyang_ssa['test_r2'][0],xianyang_vmd['test_r2'][0],xianyang_wd['test_r2'][0],]
xianyang_nrmse = [x_nrmse,xianyang_eemd['test_nrmse'][0],xianyang_ssa['test_nrmse'][0],xianyang_vmd['test_nrmse'][0],xianyang_wd['test_nrmse'][0],]
xianyang_mae = [x_mae,xianyang_eemd['test_mae'][0],xianyang_ssa['test_mae'][0],xianyang_vmd['test_mae'][0],xianyang_wd['test_mae'][0],]
xianyang_mape = [x_mape,xianyang_eemd['test_mape'][0],xianyang_ssa['test_mape'][0],xianyang_vmd['test_mape'][0],xianyang_wd['test_mape'][0],]
xianyang_ppts = [x_ppts,xianyang_eemd['test_ppts'][0],xianyang_ssa['test_ppts'][0],xianyang_vmd['test_ppts'][0],xianyang_wd['test_ppts'][0],]
xianyang_time = [x_timecost,xianyang_eemd['time_cost'][0],xianyang_ssa['time_cost'][0],xianyang_vmd['time_cost'][0],xianyang_wd['time_cost'][0],]

zhangjiashan_r2 = [z_r2,zhangjiashan_eemd['test_r2'][0],zhangjiashan_ssa['test_r2'][0],zhangjiashan_vmd['test_r2'][0],zhangjiashan_wd['test_r2'][0],]
zhangjiashan_nrmse = [z_nrmse,zhangjiashan_eemd['test_nrmse'][0],zhangjiashan_ssa['test_nrmse'][0],zhangjiashan_vmd['test_nrmse'][0],zhangjiashan_wd['test_nrmse'][0],]
zhangjiashan_mae = [z_mae,zhangjiashan_eemd['test_mae'][0],zhangjiashan_ssa['test_mae'][0],zhangjiashan_vmd['test_mae'][0],zhangjiashan_wd['test_mae'][0],]
zhangjiashan_mape = [z_mape,zhangjiashan_eemd['test_mape'][0],zhangjiashan_ssa['test_mape'][0],zhangjiashan_vmd['test_mape'][0],zhangjiashan_wd['test_mape'][0],]
zhangjiashan_ppts = [z_ppts,zhangjiashan_eemd['test_ppts'][0],zhangjiashan_ssa['test_ppts'][0],zhangjiashan_vmd['test_ppts'][0],zhangjiashan_wd['test_ppts'][0],]
zhangjiashan_time = [z_timecost,zhangjiashan_eemd['time_cost'][0],zhangjiashan_ssa['time_cost'][0],zhangjiashan_vmd['time_cost'][0],zhangjiashan_wd['time_cost'][0],]

def autolabels(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = round(height,2)
        ax.text(
            x=rect.get_x() + rect.get_width() / 2,
            y=height,
            s='{}'.format(height),
            rotation=90,
            ha='center', va='bottom',
                    )


########################################################################################
metrics_lists=[
    [huaxian_r2,xianyang_r2,zhangjiashan_r2],
    [huaxian_nrmse,xianyang_nrmse,zhangjiashan_nrmse],
    [huaxian_mae,xianyang_mae,zhangjiashan_mae],
    [huaxian_mape,xianyang_mape,zhangjiashan_mape],
    [huaxian_ppts,xianyang_ppts,zhangjiashan_ppts],
    [huaxian_time,xianyang_time,zhangjiashan_time],
]
stations=['Huaxian','Xianyang','Zhangjiashan']
pos = [2,4,6,8,10]
print(pos)
width=0.5
action=[-1,0,1]
ylims=[
    [-1.0,1.4],
    [0,2.2],
    [0,3.5],
    [0,570],
    [0,90],
    [0,3800],
]
labels=['SVR','EEMD-SVR-SUM','SSA-SVR-SUM','VMD-SVR-SUM','DWT-SVR-SUM']
y_labels=[
    r"$NSE$",r"$NRMSE(10^8m^3)$",r"$PPTS(5)(\%)$",r"$MAE(10^8m^3)$",r"$MAPE(\%)$",r"$Time(s)$"
]
fig = plt.figure(figsize=(7.48,7.48))
for i in range(len(metrics_lists)):
    ax = fig.add_subplot(3,2,i+1)
    for j in range(len(metrics_lists[i])):
        bars=ax.bar([p+action[j]*width for p in pos],metrics_lists[i][j],width,alpha=0.75,label=stations[j])
        # autolabels(bars,ax)
    # ax.set_ylim(ylims[i])
    ax.set_ylabel(y_labels[i])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels,rotation=45)
    if i==0:
        ax.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.6,1.20),
            # bbox_transform=plt.gcf().transFigure,
            ncol=3,
            shadow=False,
            frameon=True,
        )
plt.subplots_adjust(left=0.1, bottom=0.11, right=0.98,top=0.96, hspace=0.5, wspace=0.25)
plt.savefig(graphs_path+'three_stage_metrics.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'three_stage_metrics.tif',format='TIFF',dpi=600)
##############################################################################################
metrics_lists=[
    [huaxian_r2,xianyang_r2,zhangjiashan_r2],
    [huaxian_nrmse,xianyang_nrmse,zhangjiashan_nrmse],
    [huaxian_ppts,xianyang_ppts,zhangjiashan_ppts],
    [huaxian_time,xianyang_time,zhangjiashan_time],
]
stations=['Huaxian','Xianyang','Zhangjiashan']
pos = [2,4,6,8,10]
print(pos)
width=0.5
action=[-1,0,1]
ylims=[
    [-1.0,1.4],
    [0,2.2],
    [0,90],
    [0,3800],
]
labels=['SVR','EEMD-SVR-SUM','SSA-SVR-SUM','VMD-SVR-SUM','DWT-SVR-SUM']
y_labels=[
    r"$NSE$",r"$NRMSE(10^8m^3)$",r"$PPTS(5)(\%)$",r"$Time(s)$"
]
fig = plt.figure(figsize=(7.48,5))
for i in range(len(metrics_lists)):
    ax = fig.add_subplot(2,2,i+1)
    for j in range(len(metrics_lists[i])):
        bars=ax.bar([p+action[j]*width for p in pos],metrics_lists[i][j],width,alpha=0.75,label=stations[j])
        # autolabels(bars,ax)
    # ax.set_ylim(ylims[i])
    ax.set_ylabel(y_labels[i])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels,rotation=45)
    if i==0:
        ax.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.6,1.17),
            # bbox_transform=plt.gcf().transFigure,
            ncol=3,
            shadow=False,
            frameon=True,
        )
plt.subplots_adjust(left=0.08, bottom=0.16, right=0.98,top=0.95, hspace=0.55, wspace=0.25)
plt.savefig(graphs_path+'three_stage_NSE_NRMSE_PPTS_TIMECOST.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'three_stage_NSE_NRMSE_PPTS_TIMECOST.tif',format='TIFF',dpi=600)


###########################################################################################
nse_data = [
    [h_r2,x_r2,z_r2],
    [huaxian_eemd['test_r2'][0],xianyang_eemd["test_r2"][0],zhangjiashan_eemd["test_r2"][0]],
    [huaxian_ssa["test_r2"][0],xianyang_ssa["test_r2"][0],zhangjiashan_ssa["test_r2"][0]],
    [huaxian_vmd["test_r2"][0],xianyang_vmd["test_r2"][0],zhangjiashan_vmd["test_r2"][0]],
    [huaxian_wd["test_r2"][0],xianyang_wd["test_r2"][0],zhangjiashan_wd["test_r2"][0]],
]

mean_nse =[]
for i in range(len(nse_data)):
    mean_nse.append(sum(nse_data[i])/len(nse_data[i]))
for i in range(1,len(mean_nse)):
    print("Compared with SVR, mean NSE increased by {}%".format((mean_nse[i]-mean_nse[0])/mean_nse[0]*100))

nrmse_data = [
    [h_nrmse,x_nrmse,z_nrmse],
    [huaxian_eemd["test_nrmse"][0],xianyang_eemd["test_nrmse"][0],zhangjiashan_eemd["test_nrmse"][0]],
    [huaxian_ssa["test_nrmse"][0],xianyang_ssa["test_nrmse"][0],zhangjiashan_ssa["test_nrmse"][0]],
    [huaxian_vmd["test_nrmse"][0],xianyang_vmd["test_nrmse"][0],zhangjiashan_vmd["test_nrmse"][0]],
    [huaxian_wd["test_nrmse"][0],xianyang_wd["test_nrmse"][0],zhangjiashan_wd["test_nrmse"][0]],
]

mean_nrmse =[]
for i in range(len(nrmse_data)):
    mean_nrmse.append(sum(nrmse_data[i])/len(nrmse_data[i]))
for i in range(1,len(mean_nrmse)):
    print("Compared with SVR, mean NRMSE increased by {}%".format((mean_nrmse[i]-mean_nrmse[0])/mean_nrmse[0]*100))

ppts_data = [
    [h_ppts,x_ppts,z_ppts],
    [huaxian_eemd["test_ppts"][0],xianyang_eemd["test_ppts"][0],zhangjiashan_eemd["test_ppts"][0]],
    [huaxian_ssa["test_ppts"][0],xianyang_ssa["test_ppts"][0],zhangjiashan_ssa["test_ppts"][0]],
    [huaxian_vmd["test_ppts"][0],xianyang_vmd["test_ppts"][0],zhangjiashan_vmd["test_ppts"][0]],
    [huaxian_wd["test_ppts"][0],xianyang_wd["test_ppts"][0],zhangjiashan_wd["test_ppts"][0]],
]

mean_ppts =[]
for i in range(len(ppts_data)):
    mean_ppts.append(sum(ppts_data[i])/len(ppts_data[i]))
for i in range(1,len(mean_ppts)):
    print("Compared with SVR, mean PPTS increased by {}%".format((mean_ppts[i]-mean_ppts[0])/mean_ppts[0]*100))

timecost_data = [
    [h_timecost,x_timecost,z_timecost],
    [huaxian_eemd["time_cost"][0],xianyang_eemd["time_cost"][0],zhangjiashan_eemd["time_cost"][0]],
    [huaxian_ssa["time_cost"][0],xianyang_ssa["time_cost"][0],zhangjiashan_ssa["time_cost"][0]],
    [huaxian_vmd["time_cost"][0],xianyang_vmd["time_cost"][0],zhangjiashan_vmd["time_cost"][0]],
    [huaxian_wd["time_cost"][0],xianyang_wd["time_cost"][0],zhangjiashan_wd["time_cost"][0]],
]

mean_time =[]
for i in range(len(timecost_data)):
    mean_time.append(sum(timecost_data[i])/len(timecost_data[i]))
for i in range(1,len(mean_time)):
    print("Compared with SVR, mean TIME increased by {}%".format((mean_time[i]-mean_time[0])/mean_time[0]*100))

all_datas = [
    nse_data,nrmse_data,ppts_data,timecost_data
]



x = list(range(5))
x_s = [-0.38, 3.7, -0.38, 3.7]
y_s = [0.9, 1.7, 3, 212]
fig_ids = ['(a)', '(b)', '(c)', '(d)']
plt.figure(figsize=(3.54, 2.54))
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
# plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96,top=0.98, hspace=0.6, wspace=0.45)
plt.savefig(graphs_path+'/three_stage_metrics_violin.eps',
            format='EPS', dpi=2000)
plt.savefig(graphs_path+'/three_stage_metrics_violin.tif',
            format='TIFF', dpi=1200)
plt.show()
