import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size']=10
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'

font={'family':'serif',
     'style':'normal',
    'weight':'bold',
      'color':'black',
      'size':10
}


huaxian = pd.read_csv(root_path+'/Huaxian/projects/esvr/Huaxian_esvr.csv')
huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_one_month_forecast/Huaxian_eemd_one_step_esvr_forecast.csv')
huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_one_month_forecast/Huaxian_ssa_one_step_esvr_forecast.csv')
huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_one_month_forecast/Huaxian_vmd_one_step_esvr_forecast.csv')
huaxian_wd = pd.read_csv(root_path+'/Huaxian_wd/projects/esvr/db10-lev2/one_step_one_month_forecast/Huaxian_wd_one_step_esvr_forecast.csv')

plt.figure(figsize=(7.48,8))
records=huaxian_vmd['test_y'].iloc[0:120]
# plot SVR
predictions=huaxian_vmd['test_pred'].iloc[0:120]
plt.subplot(4,2,1)
plt.text(120,-13,'(a)VMD-LSTM',fontdict=font)
plt.xlabel("时间（月）")
plt.ylabel("月径流(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='实测值')
plt.plot(predictions,c='r',label='预测值')
plt.legend(
            loc='upper center',
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            bbox_to_anchor=(0.5,1.25),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=True,
            frameon=False,
            )
plt.subplot(4,2,2)
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
plt.xlabel('预测值(' + r'$10^8m^3$' +')', )
plt.ylabel('实测值(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='线性拟合',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='理想拟合',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])
plt.legend(
            loc='upper center',
            # bbox_to_anchor=(0.1,1.02, 1,0.102),
            bbox_to_anchor=(0.5,1.25),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=True,
            frameon=False,
            )


# plot EEMD-SVR
predictions=huaxian_eemd['test_pred'].iloc[0:120]
plt.subplot(4,2,3)
plt.text(120,-13,'(b)EEMD-LSTM',fontdict=font)
plt.xlabel("时间（月）")
plt.ylabel("月径流(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='实测值')
plt.plot(predictions,c='r',label='预测值')
plt.subplot(4,2,4)
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
plt.xlabel('预测值(' + r'$10^8m^3$' +')', )
plt.ylabel('实测值(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='线性拟合',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='理想拟合',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])

# plot SSA-SVR
predictions=huaxian_ssa['test_pred'].iloc[0:120]
plt.subplot(4,2,5)
plt.text(120,-13,'(c)SSA-LSTM',fontdict=font)
plt.xlabel("时间（月）")
plt.ylabel("月径流(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='实测值')
plt.plot(predictions,c='r',label='预测值')
plt.subplot(4,2,6)
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
plt.xlabel('预测值(' + r'$10^8m^3$' +')', )
plt.ylabel('实测值(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='线性拟合',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='理想拟合',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])

# plot DWT-SVR
predictions=huaxian_wd['test_pred'].iloc[0:120]
plt.subplot(4,2,7)
plt.text(120,-14,'(d)DWT-LSTM',fontdict=font)
plt.xlabel("时间（月）")
plt.ylabel("月径流(" + r"$10^8m^3$" + ")")
plt.plot(records,c='b',label='实测值')
plt.plot(predictions,c='r',label='预测值')
plt.subplot(4,2,8)
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
plt.xlabel('预测值(' + r'$10^8m^3$' +')', )
plt.ylabel('实测值(' + r'$10^8m^3$' + ')', )
plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
plt.plot(xx, linear_fit, '--', color='red', label='线性拟合',linewidth=1.0)
plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='理想拟合',linewidth=1.0)
plt.xlim([xymin,xymax])
plt.ylim([xymin,xymax])

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,top=0.96, hspace=0.4, wspace=0.25)
plt.savefig(graphs_path+'华县多模型对比.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'华县多模型对比.tif',format='TIFF',dpi=300)
plt.show()







