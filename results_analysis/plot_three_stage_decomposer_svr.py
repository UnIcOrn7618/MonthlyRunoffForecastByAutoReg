import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
print(root_path)
import sys
sys.path.append(root_path)
from results_reader import read_two_stage,read_pure_esvr
from fit_line import compute_linear_fit,compute_list_linear_fit

h_records,h_predictions,h_r2,h_nrmse,h_mae,h_mape,h_ppts,h_timecost=read_pure_esvr("Huaxian")
x_records,x_predictions,x_r2,x_nrmse,x_mae,x_mape,x_ppts,x_timecost=read_pure_esvr("Xianyang")
z_records,z_predictions,z_r2,z_nrmse,z_mae,z_mape,z_ppts,z_timecost=read_pure_esvr("Zhangjiashan")

huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_eemd_sum_test_result.csv')
huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_ssa_sum_test_result.csv')
huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_vmd_sum_test_result.csv')
huaxian_dwt = pd.read_csv(root_path+'/Huaxian_dwt/projects/esvr/db10-2/multi_step_1_month_forecast/esvr_Huaxian_dwt_sum_test_result.csv')

xianyang_eemd = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_eemd_sum_test_result.csv')
xianyang_ssa = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_ssa_sum_test_result.csv')
xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Xianyang_vmd_sum_test_result.csv')
xianyang_dwt = pd.read_csv(root_path+'/Xianyang_dwt/projects/esvr/db10-2/multi_step_1_month_forecast/esvr_Xianyang_dwt_sum_test_result.csv')

zhangjiashan_eemd = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_eemd_sum_test_result.csv')
zhangjiashan_ssa = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_ssa_sum_test_result.csv')
zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Zhangjiashan_vmd_sum_test_result.csv')
zhangjiashan_dwt = pd.read_csv(root_path+'/Zhangjiashan_dwt/projects/esvr/db10-2/multi_step_1_month_forecast/esvr_Zhangjiashan_dwt_sum_test_result.csv')




plt.figure(figsize=(7.48,4.5))
cols=9
col1=7
col2=cols-col1
ax1 = plt.subplot2grid((3,cols), (0,0), colspan=col1)
ax2 = plt.subplot2grid((3,cols), (0,col1), colspan=col2,aspect='equal')
ax3 = plt.subplot2grid((3,cols), (1,0), colspan=col1)
ax4 = plt.subplot2grid((3,cols), (1,col1), colspan=col2,aspect='equal')
ax5 = plt.subplot2grid((3,cols), (2,0), colspan=col1)
ax6 = plt.subplot2grid((3,cols), (2,col1), colspan=col2,aspect='equal')

models=['VMD-SVR-SUM','DWT-SVR-SUM','SSA-SVR-SUM','EEMD-SVR-SUM','SVR']
markers=['o','v','s','*','d']
colors=['r','g','teal','cyan','gold']
zorders=[4,3,2,1,0]

# ax1.set_xlabel('Time(month)')
ax1.set_ylabel("Flow(" + r"$10^8m^3$" + ")")
ax1.text(-5,29,'(a1)',fontweight='normal',fontsize=7)
ax1.plot(huaxian_vmd['orig'],label='Records',lw=2.5,zorder=0)
ax1.plot(huaxian_vmd['pred'],label='VMD-SVR-SUM',lw=2,zorder=1)
ax1.plot(huaxian_dwt['pred'],':',label='DWT-SVR-SUM',lw=1.5,zorder=2)
ax1.plot(huaxian_ssa['pred'],'-.',label='SSA-SVR-SUM',lw=1.0,zorder=3)
ax1.plot(huaxian_eemd['pred'],'--',label='EEMD-SVR-SUM',lw=0.5,zorder=4)
ax1.plot(h_predictions,'-',label='SVR',lw=0.5,zorder=5)
ax1.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(-0.01,1.35),
            ncol=3,
            shadow=False,
            frameon=True,
            )

records_list=[huaxian_vmd['orig'].values,huaxian_dwt['orig'].values,huaxian_ssa['orig'].values,huaxian_eemd['orig'].values,h_records]
predictions_list=[huaxian_vmd['pred'].values,huaxian_dwt['pred'].values,huaxian_ssa['pred'].values,huaxian_eemd['pred'].values,h_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
# ax2.set_xlabel('Predictions(' + r'$10^8m^3$' +')', )
ax2.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax2.text(0.6,29,'(a2)',fontweight='normal',fontsize=7)

for i in range(len(predictions_list)):
    print(predictions_list[i])
    print(records_list[i])
    # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    ax2.scatter(predictions_list[i], records_list[i],marker=markers[i],zorder=zorders[i])
    ax2.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
ax2.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
ax2.set_xlim([xymin,xymax])
ax2.set_ylim([xymin,xymax])
ax2.legend(
            loc='upper right',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(1.03,1.35),
            ncol=3,
            shadow=False,
            frameon=True,
            )


# ax3.set_xlabel('Time(month)')
ax3.set_ylabel("Flow(" + r"$10^8m^3$" + ")")
ax3.text(-5,18,'(b1)',fontweight='normal',fontsize=7)
ax3.plot(x_records,label='Records',lw=2.5,zorder=0)
ax3.plot(xianyang_vmd['pred'],label='VMD-SVR',lw=2,zorder=1)
ax3.plot(xianyang_dwt['pred'],':',label='DWT-SVR',lw=1.5,zorder=2)
ax3.plot(xianyang_ssa['pred'],'-.',label='SSA-SVR',lw=1.0,zorder=3)
ax3.plot(xianyang_eemd['pred'],'--',label='EEMD-SVR',lw=0.5,zorder=4)
ax3.plot(x_predictions,'-',label='SVR',lw=0.5,zorder=5)

records_list=[xianyang_vmd['orig'].values,xianyang_dwt['orig'].values,xianyang_ssa['orig'].values,xianyang_eemd['orig'].values,x_records]
predictions_list=[xianyang_vmd['pred'].values,xianyang_dwt['pred'].values,xianyang_ssa['pred'].values,xianyang_eemd['pred'].values,x_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
# ax4.set_xlabel('Predictions(' + r'$10^8m^3$' +')', )
ax4.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax4.text(0.5,17.5,'(b2)',fontweight='normal',fontsize=7)
for i in range(len(predictions_list)):
    print(predictions_list[i])
    print(records_list[i])
    # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    ax4.scatter(predictions_list[i], records_list[i],marker=markers[i], label=models[i],zorder=zorders[i])
    ax4.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
ax4.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
ax4.set_xlim([xymin,xymax])
ax4.set_ylim([xymin,xymax])


ax5.set_xlabel('Time(2009/01-2018/12)')
ax5.set_ylabel("Flow(" + r"$10^8m^3$" + ")")
ax5.text(-5,5.8,'(c1)',fontweight='normal',fontsize=7)
ax5.plot(z_records,label='Records',lw=2.5,zorder=0)
ax5.plot(zhangjiashan_vmd['pred'],label='VMD-SVR-SUM',lw=2,zorder=1)
ax5.plot(zhangjiashan_dwt['pred'],':',label='DWT-SVR-SUM',lw=1.5,zorder=2)
ax5.plot(zhangjiashan_ssa['pred'],'-.',label='SSA-SVR-SUM',lw=1.0,zorder=3)
ax5.plot(zhangjiashan_eemd['pred'],'--',label='EEMD-SVR-SUM',lw=0.5,zorder=4)
ax5.plot(z_predictions,'-',label='SVR',lw=0.5,zorder=5)

records_list=[zhangjiashan_vmd['orig'].values,zhangjiashan_dwt['orig'].values,zhangjiashan_ssa['orig'].values,zhangjiashan_eemd['orig'].values,z_records]
predictions_list=[zhangjiashan_vmd['pred'].values,zhangjiashan_dwt['pred'].values,zhangjiashan_ssa['pred'].values,zhangjiashan_eemd['pred'].values,z_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
ax6.set_xlabel('Predictions(' + r'$10^8m^3$' +')', )
ax6.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax6.text(0.25,5.8,'(c2)',fontweight='normal',fontsize=7)
for i in range(len(predictions_list)):
    print(predictions_list[i])
    print(records_list[i])
    # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    ax6.scatter(predictions_list[i], records_list[i],marker=markers[i], label=models[i],zorder=zorders[i])
    ax6.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
ax6.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
ax6.set_xlim([xymin,xymax])
ax6.set_ylim([xymin,xymax])

# plt.tight_layout()
plt.subplots_adjust(left=0.06, bottom=0.09, right=0.99,top=0.92, hspace=0.2, wspace=1.0)
plt.savefig(graphs_path+'three_stage_decomposer_svr.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'three_stage_decomposer_svr.tif',format='TIFF',dpi=600)
plt.show()

