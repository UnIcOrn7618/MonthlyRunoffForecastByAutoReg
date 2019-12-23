import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
print(root_path)
import sys
sys.path.append(root_path+'/tools/')
from results_reader import read_two_stage,read_pure_esvr
from fit_line import compute_linear_fit,compute_list_linear_fit

h_records,h_predictions,h_r2,h_nrmse,h_mae,h_mape,h_ppts,h_timecost=read_pure_esvr("Huaxian")
x_records,x_predictions,x_r2,x_nrmse,x_mae,x_mape,x_ppts,x_timecost=read_pure_esvr("Xianyang")
z_records,z_predictions,z_r2,z_nrmse,z_mae,z_mape,z_ppts,z_timecost=read_pure_esvr("Zhangjiashan")

h_vmd_records,h_vmd_predictions,h_vmd_r2,h_vmd_nrmse,h_vmd_mae,h_vmd_mape,h_vmd_ppts,h_vmd_timecost= read_two_stage(station="Huaxian",decomposer="vmd",predict_pattern="one_step_1_month_forecast")
x_vmd_records,x_vmd_predictions,x_vmd_r2,x_vmd_nrmse,x_vmd_mae,x_vmd_mape,x_vmd_ppts,x_vmd_timecost= read_two_stage(station="Xianyang",decomposer="vmd",predict_pattern="one_step_1_month_forecast")
z_vmd_records,z_vmd_predictions,z_vmd_r2,z_vmd_nrmse,z_vmd_mae,z_vmd_mape,z_vmd_ppts,z_vmd_timecost= read_two_stage(station="Zhangjiashan",decomposer="vmd",predict_pattern="one_step_1_month_forecast")

h_eemd_records,h_eemd_predictions,h_eemd_r2,h_eemd_nrmse,h_eemd_mae,h_eemd_mape,h_eemd_ppts,h_eemd_timecost= read_two_stage(station="Huaxian",decomposer="eemd",predict_pattern="one_step_1_month_forecast")
x_eemd_records,x_eemd_predictions,x_eemd_r2,x_eemd_nrmse,x_eemd_mae,x_eemd_mape,x_eemd_ppts,x_eemd_timecost= read_two_stage(station="Xianyang",decomposer="eemd",predict_pattern="one_step_1_month_forecast")
z_eemd_records,z_eemd_predictions,z_eemd_r2,z_eemd_nrmse,z_eemd_mae,z_eemd_mape,z_eemd_ppts,z_eemd_timecost= read_two_stage(station="Zhangjiashan",decomposer="eemd",predict_pattern="one_step_1_month_forecast")

h_ssa_records,h_ssa_predictions,h_ssa_r2,h_ssa_nrmse,h_ssa_mae,h_ssa_mape,h_ssa_ppts,h_ssa_timecost= read_two_stage(station="Huaxian",decomposer="ssa",predict_pattern="one_step_1_month_forecast")
x_ssa_records,x_ssa_predictions,x_ssa_r2,x_ssa_nrmse,x_ssa_mae,x_ssa_mape,x_ssa_ppts,x_ssa_timecost= read_two_stage(station="Xianyang",decomposer="ssa",predict_pattern="one_step_1_month_forecast")
z_ssa_records,z_ssa_predictions,z_ssa_r2,z_ssa_nrmse,z_ssa_mae,z_ssa_mape,z_ssa_ppts,z_ssa_timecost= read_two_stage(station="Zhangjiashan",decomposer="ssa",predict_pattern="one_step_1_month_forecast")

h_wd_records,h_wd_predictions,h_wd_r2,h_wd_nrmse,h_wd_mae,h_wd_mape,h_wd_ppts,h_wd_timecost= read_two_stage(station="Huaxian",decomposer="wd",predict_pattern="one_step_1_month_forecast")
x_wd_records,x_wd_predictions,x_wd_r2,x_wd_nrmse,x_wd_mae,x_wd_mape,x_wd_ppts,x_wd_timecost= read_two_stage(station="Xianyang",decomposer="wd",predict_pattern="one_step_1_month_forecast")
z_wd_records,z_wd_predictions,z_wd_r2,z_wd_nrmse,z_wd_mae,z_wd_mape,z_wd_ppts,z_wd_timecost= read_two_stage(station="Zhangjiashan",decomposer="wd",predict_pattern="one_step_1_month_forecast")

huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_eemd_sum_test_result.csv')
huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_ssa_sum_test_result.csv')
huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_vmd_sum_test_result.csv')
huaxian_wd = pd.read_csv(root_path+'/Huaxian_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Huaxian_wd_sum_test_result.csv')

xianyang_eemd = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_eemd_sum_test_result.csv')
xianyang_ssa = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_ssa_sum_test_result.csv')
xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_vmd_sum_test_result.csv')
xianyang_wd = pd.read_csv(root_path+'/Xianyang_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Xianyang_wd_sum_test_result.csv')

zhangjiashan_eemd = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_eemd_sum_test_result.csv')
zhangjiashan_ssa = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_ssa_sum_test_result.csv')
zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_vmd_sum_test_result.csv')
zhangjiashan_wd = pd.read_csv(root_path+'/Zhangjiashan_wd/projects/esvr/db10-lev2/multi_step_1_month_forecast/esvr_Zhangjiashan_wd_sum_test_result.csv')


##########################################################################################################
records_list=[
    [h_eemd_records,huaxian_eemd['orig'].values,],
    [h_ssa_records,huaxian_ssa['orig'].values,],
    [h_vmd_records,huaxian_vmd['orig'].values,],
    [h_wd_records,huaxian_wd['orig'].values,],
    [x_eemd_records,xianyang_eemd['orig'].values,],
    [x_ssa_records,xianyang_ssa['orig'].values,],
    [x_vmd_records,xianyang_vmd['orig'].values,],
    [x_wd_records,xianyang_wd['orig'].values,],
    [z_eemd_records,zhangjiashan_eemd['orig'].values,],
    [z_ssa_records,zhangjiashan_ssa['orig'].values,],
    [z_vmd_records,zhangjiashan_vmd['orig'].values,],
    [z_wd_records,zhangjiashan_wd['orig'].values,],
  
    
]
predictions_list=[
    [h_eemd_predictions,huaxian_eemd['pred'].values,],
    [h_ssa_predictions,huaxian_ssa['pred'].values,],
    [h_vmd_predictions,huaxian_vmd['pred'].values,],
    [h_wd_predictions,huaxian_wd['pred'].values,],
    [x_eemd_predictions,xianyang_eemd['pred'].values,],
    [x_ssa_predictions,xianyang_ssa['pred'].values,],
    [x_vmd_predictions,xianyang_vmd['pred'].values,],
    [x_wd_predictions,xianyang_wd['pred'].values,],
    [z_eemd_predictions,zhangjiashan_eemd['pred'].values,],
    [z_ssa_predictions,zhangjiashan_ssa['pred'].values,],
    [z_vmd_predictions,zhangjiashan_vmd['pred'].values,],
    [z_wd_predictions,zhangjiashan_wd['pred'].values,],

]
fig_id=[
    '(a1)','(a2)','(a3)','(a4)',
    '(b1)','(b2)','(b3)','(b4)',
    '(c1)','(c2)','(c3)','(c4)',
]
models_labels=[
    ['EEMD-SVR','EEMD-SVR-SUM',],
    ['SSA-SVR','SSA-SVR-SUM',],
    ['VMD-SVR','VMD-SVR-SUM',],
    ['DWT-SVR','DWT-SVR-SUM',],
    ['EEMD-SVR','EEMD-SVR-SUM',],
    ['SSA-SVR','SSA-SVR-SUM',],
    ['VMD-SVR','VMD-SVR-SUM',],
    ['DWT-SVR','DWT-SVR-SUM',],
    ['EEMD-SVR','EEMD-SVR-SUM',], 
    ['SSA-SVR','SSA-SVR-SUM',],
    ['VMD-SVR','VMD-SVR-SUM',],
    ['DWT-SVR','DWT-SVR-SUM',],
]



x_s=[27.5,27.5,27.5,27.7,16.2,16.5,16.5,17.5,5.5,4.28,4.25,4.35,]
y_s=[1.5,2,1.85,2,0.9,1.0,1.0,1.1,0.24,0.18,0.18,0.2,]

plt.figure(figsize=(7.48,5.5))
for j in range(len(records_list)):
    ax=plt.subplot(3,4,j+1, aspect='equal')
    xx,linear_list,xymin,xymax=compute_list_linear_fit(
        records_list=records_list[j],
        predictions_list=predictions_list[j],
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    if j in range(8,12):
        plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
    if j in [0,4,8]:
        plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
    models=models_labels[j]
    markers=['o','v',]
    zorders=[1,0]
    plt.text(x_s[j],y_s[j],fig_id[j],fontweight='normal',fontsize=7)
    for i in range(len(predictions_list[j])):
        # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
        plt.scatter(predictions_list[j][i], records_list[j][i],marker=markers[i],zorder=zorders[i])
        plt.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    plt.legend(
                loc=0,
                # bbox_to_anchor=(0.08,1.01, 1,0.101),
                # bbox_to_anchor=(1,1),
                # ncol=2,
                shadow=False,
                frameon=False,
                fontsize=6,
                )

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.99,top=0.99, hspace=0.15, wspace=0.1)
plt.savefig(graphs_path+'two_three_stage_decomposer_scatter.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'two_three_stage_decomposer_scatter.tif',format='TIFF',dpi=600)
plt.show()

