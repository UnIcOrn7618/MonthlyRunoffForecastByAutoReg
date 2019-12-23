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

models=['VMD-SVR','DWT-SVR','SSA-SVR','EEMD-SVR','SVR']
markers=['o','v','s','*','d']
colors=['r','g','teal','cyan','gold']
zorders=[4,3,2,1,0]

# ax1.set_xlabel('Time(month)\n(a1)')
ax1.set_ylabel("Flow(" + r"$10^8m^3$" + ")")
ax1.text(-5,30,'(a1)',fontweight='normal',fontsize=7)
ax1.plot(h_records,label='Records',lw=2.5,zorder=0)
ax1.plot(h_vmd_predictions,label='VMD-SVR',lw=2,zorder=1)
ax1.plot(h_wd_predictions,':',label='DWT-SVR',lw=1.5,zorder=2)
ax1.plot(h_ssa_predictions,'-.',label='SSA-SVR',lw=1.0,zorder=3)
ax1.plot(h_eemd_predictions,'--',label='EEMD-SVR',lw=0.5,zorder=4)
ax1.plot(h_predictions,'-',label='SVR',lw=0.5,zorder=4)
ax1.legend(
            loc='upper left',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(-0.01,1.35),
            ncol=3,
            shadow=False,
            frameon=True,
            )

records_list=[h_vmd_records,h_wd_records,h_ssa_records,h_eemd_records,h_records]
predictions_list=[h_vmd_predictions,h_wd_predictions,h_ssa_predictions,h_eemd_predictions,h_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
# ax2.set_xlabel('Predictions(' + r'$10^8m^3$' +')\n(a2)', )
ax2.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax2.text(25.9,2,'(a2)',fontweight='normal',fontsize=7)
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


# ax3.set_xlabel('Time(month)\n(b1)')
ax3.set_ylabel("Flow(" + r"$10^8m^3$" + ")")
ax3.text(-5,19,'(b1)',fontweight='normal',fontsize=7)
ax3.plot(x_records,label='Records',lw=2.5,zorder=0)
ax3.plot(x_vmd_predictions,label='VMD-SVR',lw=2,zorder=1)
ax3.plot(x_wd_predictions,':',label='DWT-SVR',lw=1.5,zorder=2)
ax3.plot(x_ssa_predictions,'-.',label='SSA-SVR',lw=1.0,zorder=3)
ax3.plot(x_eemd_predictions,'--',label='EEMD-SVR',lw=0.5,zorder=4)
ax3.plot(x_predictions,'-',label='SVR',lw=0.5,zorder=5)

records_list=[x_vmd_records,x_wd_records,x_ssa_records,x_eemd_records,x_records]
predictions_list=[x_vmd_predictions,x_wd_predictions,x_ssa_predictions,x_eemd_predictions,x_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
# ax4.set_xlabel('Predictions(' + r'$10^8m^3$' +')\n(b2)', )
ax4.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax4.text(16.2,1,'(b2)',fontweight='normal',fontsize=7)

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
ax5.text(-5,4.5,'(c1)',fontweight='normal',fontsize=7)
ax5.plot(z_records,label='Records',lw=2.5,zorder=0)
ax5.plot(z_vmd_predictions,label='VMD-SVR',lw=2,zorder=1)
ax5.plot(z_wd_predictions,':',label='DWT-SVR',lw=1.5,zorder=2)
ax5.plot(z_ssa_predictions,'-.',label='SSA-SVR',lw=1.0,zorder=3)
ax5.plot(z_eemd_predictions,'--',label='EEMD-SVR',lw=0.5,zorder=4)
ax5.plot(z_predictions,'-',label='SVR',lw=0.5,zorder=5)

records_list=[z_vmd_records,z_wd_records,z_ssa_records,z_eemd_records,z_records]
predictions_list=[z_vmd_predictions,z_wd_predictions,z_ssa_predictions,z_eemd_predictions,z_predictions]
xx,linear_list,xymin,xymax=compute_list_linear_fit(
    records_list=records_list,
    predictions_list=predictions_list,
)
ax6.set_xlabel('Predictions(' + r'$10^8m^3$' +')', )
ax6.set_ylabel('Records(' + r'$10^8m^3$' + ')', )
ax6.text(3.9,0.3,'(c2)',fontweight='normal',fontsize=7)
for i in range(len(predictions_list)):
    print(predictions_list[i])
    print(records_list[i])
    # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    ax6.scatter(predictions_list[i], records_list[i],marker=markers[i], label=models[i],zorder=zorders[i])
    ax6.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
ax6.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
ax6.set_xlim([xymin,xymax])
ax6.set_ylim([xymin,xymax])


plt.subplots_adjust(left=0.06, bottom=0.09, right=0.99,top=0.92, hspace=0.2, wspace=1.0)
plt.savefig(graphs_path+'two_stage_decomposer_svr.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'two_stage_decomposer_svr.tif',format='TIFF',dpi=600)
plt.show()

