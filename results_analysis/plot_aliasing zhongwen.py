import matplotlib.pyplot as plt
plt.rcParams['font.size']=10
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size']=10
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

huxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/data/VMD_TRAIN_K9.csv')
xianyang_vmd = pd.read_csv(root_path+'/Xianyang_vmd/data/VMD_TRAIN_K9.csv')
zhangjiashan_vmd=pd.read_csv(root_path+'/Zhangjiashan_vmd/data/VMD_TRAIN_K8.csv')

T=huxian_vmd.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T

plt.figure(figsize=(5.5139,3.5417))
plt.subplot(2,2,1)
# plt.xlabel('Time(month)',fontsize=10)
plt.ylabel('IMF8',fontsize=10)
plt.plot(huxian_vmd['IMF8'],color='b',label='',linewidth=0.8)

plt.subplot(2,2,2)
# plt.title('IMF8',fontsize=10,loc='left')
plt.plot(freqs,abs(fft(huxian_vmd['IMF8'])),c='b',lw=0.8)
plt.ylabel('振幅')

plt.subplot(2,2,3)
plt.xlabel('时间(月)',fontsize=10)
plt.ylabel('IMF9',fontsize=10)
plt.plot(huxian_vmd['IMF9'],color='b',label='',linewidth=0.8)

plt.subplot(2,2,4)
# plt.title('IMF9',fontsize=10,loc='left')
plt.plot(freqs,abs(fft(huxian_vmd['IMF9'])),c='b',lw=0.8)
plt.vlines(x=-0.025,ymin=30,ymax=95,lw=1.5,colors='r')
plt.vlines(x=0.025,ymin=30,ymax=95,lw=1.5,colors='r')
plt.hlines(y=30,xmin=-0.025,xmax=0.025,lw=1.5,colors='r')
plt.hlines(y=95,xmin=-0.025,xmax=0.025,lw=1.5,colors='r')
plt.ylabel('振幅')
plt.xlabel('频率')

plt.tight_layout()
plt.savefig(graphs_path+'/huxian_vmd_aliasing中文版.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/huxian_vmd_aliasing中文版.tif',format='TIFF',dpi=1000)
plt.show()