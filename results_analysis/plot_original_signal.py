import matplotlib.pyplot as plt
plt.rcParams['font.size']=10
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from collections import OrderedDict
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

import sys
sys.path.append(root_path+'/tools/')
from mann_kendall import plot_trend

huaxian = pd.read_excel(root_path+'/time_series/HuaxianRunoff1951-2018(1953-2018).xlsx')['MonthlyRunoff'][24:]
xianyang = pd.read_excel(root_path+'/time_series/XianyangRunoff1951-2018(1953-2018).xlsx')['MonthlyRunoff'][24:]
zhangjiashan = pd.read_excel(root_path+'/time_series/ZhangjiashanRunoff1953-2018(1953-2018).xlsx')['MonthlyRunoff'][0:]

# dates=["1953-01-01","2019-01-01"]
# start,end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
# m=list(OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys())
# print(len(m))
fig=plt.figure(figsize=(5.5139,6.2))
ax1=fig.add_subplot(3,1,1)
ax2=fig.add_subplot(3,1,2)
ax3=fig.add_subplot(3,1,3)
plot_trend(huaxian,ax=ax1,series_name="Huaxian",fig_id="a")
plot_trend(xianyang,ax=ax2,series_name="Xianyang",fig_id="b")
plot_trend(zhangjiashan,ax=ax3,series_name="Zhangjiashan",fig_id="c")
ax1.set_xlabel("Time(month)\n(a)")
ax2.set_xlabel("Time(month)\n(b)")
ax3.set_xlabel("Time(month)\n(c)")
plt.tight_layout()
plt.savefig(graphs_path+'/original_series.EPS',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/original_series.tif',format='TIFF',dpi=1000)
plt.show()