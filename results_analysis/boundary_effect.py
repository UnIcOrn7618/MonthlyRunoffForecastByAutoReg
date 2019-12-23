import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from datetime import datetime,timedelta
from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
vmd_full = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_FULL.csv")
subsignal="IMF1"


dates=["1953-01-01","2019-01-01"]
start,end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
m=list(OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys())
print((m[529:560]))

test_imf = []
for i in range(553,792+1):
    data=pd.read_csv(root_path+"/Huaxian_vmd/data/vmd-test/vmd_appended_test"+str(i)+".csv")
    test_imf.append((data[subsignal].iloc[data.shape[0]-1:]).values.flatten()[0])
t_e=list(range(1,793))
t_t=list(range(1,553))

plt.figure(figsize=(3.54,3.0))
plt.subplot(2,1,1)
plt.plot(t_t,vmd_train[subsignal],c='r',lw=2,label="Concurrent decomposition of training set")
plt.plot(t_e,vmd_full[subsignal],c='b',label="Concurrent decomposition of entire streamflow")
plt.xlabel("Time(1997/02-1999/08)")
plt.text(558,2.08,'(a)',fontsize=7,fontweight='normal')
plt.ylabel(r"Runoff($10^8m^3$)")
plt.xlim([530,560])
plt.ylim([2,4])
plt.legend()

plt.subplot(2,1,2)
t=list(range(553,793))
print(t)
print(test_imf)
plt.plot(t,test_imf,c='r',lw=2,label="Sequential decomposition of validation set")
plt.plot(t,vmd_full[subsignal].iloc[vmd_full.shape[0]-240:],c='b',label="Concurrent decomposition of entire streamflow")
plt.xlabel("Time(1999/01-2018/12)")
plt.text(786,0.48,'(b)',fontsize=7,fontweight='normal')
plt.ylabel(r"Runoff($10^8m^3$)")
# plt.xlim([550,560])
plt.ylim([0,10])
plt.legend()
plt.tight_layout()
# plt.subplots_adjust(left=0.09, bottom=0.06, right=0.98,top=0.96, hspace=0.4, wspace=0.3)
plt.savefig(graphs_path+'/huaxian_vmd_boundary_effect.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/huaxian_vmd_boundary_effect.tif',format='TIFF',dpi=600)
plt.show()
