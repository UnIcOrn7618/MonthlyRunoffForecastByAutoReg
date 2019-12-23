import matplotlib.pyplot as plt
plt.rcParams['font.size']=10
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'
import sys
sys.path.append(root_path+'/tools/')
from results_reader import read_two_stage

h_records,h_predictions,h_r2,h_rmse,h_mae,h_mape,h_ppts,h_timecost= read_two_stage(station="Huaxian",decomposer="vmd",predict_pattern="one_step_1_month_forecast")
x_records,x_predictions,x_r2,x_rmse,x_mae,x_mape,x_ppts,x_timecost= read_two_stage(station="Xianyang",decomposer="vmd",predict_pattern="one_step_1_month_forecast")
z_records,z_predictions,z_r2,z_rmse,z_mae,z_mape,z_ppts,z_timecost= read_two_stage(station="Zhangjiashan",decomposer="vmd",predict_pattern="one_step_1_month_forecast")


plt.figure(figsize=(7.48,7.48))
# plot predictions for huaxian station
records=h_records
predictions=h_predictions
plt.subplot(3,2,1)
plt.text(138,-12,'(a)')
# plt.title('Huaxian',fontsize=10,loc='left')
plt.xlabel('Time(month)')
plt.ylabel("flow(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='records')
plt.plot(predictions,c='r',label='predictions')
plt.legend(
            loc='upper center',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.5,1.2),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=False,
            frameon=True,
            )
plt.subplot(3,2,2)
pred_min =predictions.min()
pred_max = predictions.max()
record_min = records.min()
record_max = records.max()
if pred_min<record_min:
    xymin = pred_min
else:
    xymin = record_min
if pred_max>record_max:
    xymax = pred_max
else:
    xymax=record_max
xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
coeff = np.polyfit(predictions, records, 1)
linear_fit = coeff[0] * xx + coeff[1]
plt.xlabel('predictions(' + r'$10^8m^3$' +')', )
plt.ylabel('records(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])

plt.legend(
            loc='upper center',
            # bbox_to_anchor=(0.1,1.02, 1,0.102),
            bbox_to_anchor=(0.5,1.2),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=False,
            frameon=True,
            )

# plot predictions for xianyang station
records=x_records
predictions=x_predictions
plt.subplot(3,2,3)
plt.text(138,-8,'(b)')
# plt.title('Xianyang',fontsize=10,loc='left')
plt.xlabel('Time(month)')
plt.ylabel("flow(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='records')
plt.plot(predictions,c='r',label='predictions')

plt.subplot(3,2,4)
pred_min =predictions.min()
pred_max = predictions.max()
record_min = records.min()
record_max = records.max()
if pred_min<record_min:
    xymin = pred_min
else:
    xymin = record_min
if pred_max>record_max:
    xymax = pred_max
else:
    xymax=record_max
xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
coeff = np.polyfit(predictions, records, 1)
linear_fit = coeff[0] * xx + coeff[1]
plt.xlabel('predictions(' + r'$10^8m^3$' +')', )
plt.ylabel('records(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])


# plot predictions for zhangjiashan station
records=z_records
predictions=z_predictions
plt.subplot(3,2,5)
plt.text(138,-2,'(c)')
# plt.title('Zhangjiashan',fontsize=10,loc='left')
plt.xlabel('Time(month)')
plt.ylabel("flow(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='records')
plt.plot(predictions,c='r',label='predictions')

plt.subplot(3,2,6)
pred_min =predictions.min()
pred_max = predictions.max()
record_min = records.min()
record_max = records.max()
if pred_min<record_min:
    xymin = pred_min
else:
    xymin = record_min
if pred_max>record_max:
    xymax = pred_max
else:
    xymax=record_max
xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
coeff = np.polyfit(predictions, records, 1)
linear_fit = coeff[0] * xx + coeff[1]
plt.xlabel('predictions(' + r'$10^8m^3$' +')', )
plt.ylabel('records(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])


# plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,top=0.96, hspace=0.4, wspace=0.25)
plt.savefig(graphs_path+'two_stage_vmd_svr.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'two_stage_vmd_svr.tif',format='TIFF',dpi=1000)
plt.show()

