import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6

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

plt.figure(figsize=(3.54,2.0))
ax1=plt.subplot(2,2,1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# plt.xlabel('Time(month)')
plt.ylabel(r'$S_8$')
plt.plot(huxian_vmd['IMF8'],color='b',label='',linewidth=0.8)

plt.subplot(2,2,2)
# plt.title('IMF8',loc='left')
plt.plot(freqs,abs(fft(huxian_vmd['IMF8'])),c='b',lw=0.8)
plt.ylabel('Amplitude')

plt.subplot(2,2,3)
plt.xlabel('Time(month)')
plt.ylabel(r'$S_9$')
plt.plot(huxian_vmd['IMF9'],color='b',label='',linewidth=0.8)

plt.subplot(2,2,4)
# plt.title('IMF9',loc='left')
plt.plot(freqs,abs(fft(huxian_vmd['IMF9'])),c='b',lw=0.8,zorder=0)
plt.vlines(x=-0.025,ymin=30,ymax=95,lw=1.5,colors='r',zorder=1)
plt.vlines(x=0.025,ymin=30,ymax=95,lw=1.5,colors='r',zorder=1)
plt.hlines(y=30,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)
plt.hlines(y=95,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)
plt.ylabel('Amplitude')
plt.xlabel('Frequency(1/month)')

plt.tight_layout()
plt.savefig(graphs_path+'/huxian_vmd_aliasing.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/huxian_vmd_aliasing.tif',format='TIFF',dpi=1200)
# plt.show()


plt.figure(figsize=(7.48,3.0))
plt.subplot(2,3,1)
plt.title("IMF8",loc="left")
plt.plot(freqs,abs(fft(huxian_vmd['IMF8'])),c='b',lw=0.8)
plt.ylabel('Amplitude')

print("Frequency:{}".format(freqs))
print("Amplitude:{}".format(abs(fft(huxian_vmd['IMF8']))))

plt.subplot(2,3,4)
plt.title("IMF9",loc="left")
plt.xlabel('Frequency(1/month)\n(a)')
plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(huxian_vmd['IMF9'])),c='b',lw=0.8,zorder=0)
plt.vlines(x=-0.025,ymin=30,ymax=95,lw=1.5,colors='r',zorder=1)
plt.vlines(x=0.025,ymin=30,ymax=95,lw=1.5,colors='r',zorder=1)
plt.hlines(y=30,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)
plt.hlines(y=95,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)


plt.subplot(2,3,2)
plt.title("IMF8",loc="left")
plt.plot(freqs,abs(fft(xianyang_vmd['IMF8'])),c='b',lw=0.8)
plt.ylabel('Amplitude')

plt.subplot(2,3,5)
plt.title("IMF9",loc="left")
plt.xlabel('Frequency(1/month)\n(b)')
plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(xianyang_vmd['IMF9'])),c='b',lw=0.8,zorder=0)
plt.vlines(x=-0.025,ymin=3,ymax=40,lw=1.5,colors='r',zorder=1)
plt.vlines(x=0.025,ymin=3,ymax=40,lw=1.5,colors='r',zorder=1)
plt.hlines(y=3,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)
plt.hlines(y=40,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)


plt.subplot(2,3,3)
plt.title("IMF7",loc="left")
plt.plot(freqs,abs(fft(zhangjiashan_vmd['IMF7'])),c='b',lw=0.8)
plt.ylabel('Amplitude')

plt.subplot(2,3,6)
plt.title("IMF8",loc="left")
plt.xlabel('Frequency(1/month)\n(c)')
plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(zhangjiashan_vmd['IMF8'])),c='b',lw=0.8,zorder=0)
plt.vlines(x=-0.025,ymin=3,ymax=27,lw=1.5,colors='r',zorder=1)
plt.vlines(x=0.025,ymin=3,ymax=27,lw=1.5,colors='r',zorder=1)
plt.hlines(y=3,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)
plt.hlines(y=30,xmin=-0.025,xmax=0.025,lw=1.5,colors='r',zorder=1)


plt.tight_layout()
plt.savefig(graphs_path+'/vmd_aliasing.eps',format='EPS',dpi=2000)
plt.savefig(graphs_path+'/vmd_aliasing.tif',format='TIFF',dpi=600)
plt.show()