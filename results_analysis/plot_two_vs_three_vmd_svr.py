import matplotlib.pyplot as plt
plt.rcParams['font.size']=10
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'

# huaxian = pd.read_csv(root_path+'/Huaxian/projects/esvr/Huaxian_esvr.csv')
# two_huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/one_step_one_month_forecast/Huaxian_eemd_one_step_esvr_forecast.csv')
# two_huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/one_step_one_month_forecast/Huaxian_ssa_one_step_esvr_forecast.csv')
two_huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/one_step_one_month_forecast/Huaxian_vmd_one_step_esvr_forecast.csv')
# two_huaxian_wd = pd.read_csv(root_path+'/Huaxian_wd/projects/esvr/db10-lev2/one_step_one_month_forecast/Huaxian_wd_one_step_esvr_forecast.csv')
# three_huaxian_eemd = pd.read_csv(root_path+'/Huaxian_eemd/projects/esvr/multi_step_one_month_forecast/esvr_Huaxian_eemd_sum_test_result.csv')
# three_huaxian_ssa = pd.read_csv(root_path+'/Huaxian_ssa/projects/esvr/multi_step_one_month_forecast/esvr_Huaxian_ssa_sum_test_result.csv')
three_huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/projects/esvr/multi_step_one_month_forecast/esvr_Huaxian_vmd_sum_test_result.csv')
three_huaxian_wd = pd.read_csv(root_path+'/Huaxian_wd/projects/esvr/db10-lev2/multi_step_one_month_forecast/esvr_Huaxian_wd_sum_test_result.csv')

# two_xianyang = pd.read_csv(root_path+'/Xianyang/projects/esvr/Xianyang_esvr.csv')
# two_xianyang_eemd = pd.read_csv(root_path+'/Xianyang_eemd/projects/esvr/one_step_one_month_forecast/Xianyang_eemd_one_step_esvr_forecast.csv')
# two_xianyang_ssa = pd.read_csv(root_path+'/Xianyang_ssa/projects/esvr/one_step_one_month_forecast/Xianyang_ssa_one_step_esvr_forecast.csv')
two_xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/one_step_one_month_forecast/Xianyang_vmd_one_step_esvr_forecast.csv')
# two_xianyang_wd = pd.read_csv(root_path+'/Xianyang_wd/projects/esvr/db10-lev2/one_step_one_month_forecast/Xianyang_wd_one_step_esvr_forecast.csv')
three_xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/projects/esvr/multi_step_one_month_forecast/esvr_Xianyang_vmd_sum_test_result.csv')

# two_zhangjiashan=pd.read_csv(root_path+'/Zhangjiashan/projects/esvr/Zhangjiashan_esvr.csv')
# two_zhangjiashan_eemd = pd.read_csv(root_path+'/Zhangjiashan_eemd/projects/esvr/one_step_one_month_forecast/Zhangjiashan_eemd_one_step_esvr_forecast.csv')
# two_zhangjiashan_ssa = pd.read_csv(root_path+'/Zhangjiashan_ssa/projects/esvr/one_step_one_month_forecast/Zhangjiashan_ssa_one_step_esvr_forecast.csv')
two_zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/one_step_one_month_forecast/Zhangjiashan_vmd_one_step_esvr_forecast.csv')
# two_zhangjiashan_wd = pd.read_csv(root_path+'/Zhangjiashan_wd/projects/esvr/db10-lev2/one_step_one_month_forecast/Zhangjiashan_wd_one_step_esvr_forecast.csv')
three_zhangjiashan_vmd = pd.read_csv(root_path+'/Zhangjiashan_vmd/projects/esvr/multi_step_one_month_forecast/esvr_Zhangjiashan_vmd_sum_test_result.csv')


plt.figure(figsize=(7.48,7.48))
plt.subplot(3,1,1)
plt.text(138,-13,'(a)')
plt.xlabel("Time(month)")
plt.ylabel("flow(" + r"$10^8m^3$" + ")")
plt.plot(two_huaxian_vmd['test_y'].iloc[0:120],c='b',label='records')
plt.plot(two_huaxian_vmd['test_pred'].iloc[0:120],c='r',label='VMD-SVR')
plt.plot(three_huaxian_vmd['pred'].iloc[0:120],c='g',label='VMD-SVR-SUM')
plt.legend(
            loc=0,
            # bbox_to_anchor=(0.08,1.01, 1,0.101),
            # bbox_to_anchor=(0.5,1.25),
            # bbox_transform=plt.gcf().transFigure,
            # ncol=3,
            shadow=False,
            frameon=True,
            )


plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,top=0.96, hspace=0.4, wspace=0.25)
plt.savefig(graphs_path+'two_vs_three_stage_vmd_svr.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'two_vs_three_stage_vmd_svr.tif',format='TIFF',dpi=1000)
plt.show()





