import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import math
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

huaxian_vmd = pd.read_csv(root_path+'/Huaxian_vmd/data/VMD_TRAIN.csv')


T=huaxian_vmd.shape[0] #sampling frequency
fs=1/T #sampling period(interval)
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T #sampling times
freqs = t-0.5-1/T
L = huaxian_vmd.shape[1]-1
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('IMF'+str(i))
    plt.plot(huaxian_vmd['IMF'+str(i)],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_vmd['IMF'+str(i)])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()

# plt.savefig(graphs_path+'/vmd_aliasing.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'/vmd_aliasing.tif',format='TIFF',dpi=600)


huaxian_eemd = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
T=huaxian_eemd.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_eemd.shape[1]-1
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('IMF'+str(i))
    plt.plot(huaxian_eemd['IMF'+str(i)],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_eemd['IMF'+str(i)])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()
# plt.savefig(graphs_path+'/vmd_aliasing.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'/vmd_aliasing.tif',format='TIFF',dpi=600)

huaxian_ssa = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
T=huaxian_ssa.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_ssa.shape[1]-1
columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise']
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel('S'+str(i))
    plt.plot(huaxian_ssa[columns[i-1]],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_ssa[columns[i-1]])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Hz)')
    plt.ylabel('Amplitude')
plt.tight_layout()
# plt.savefig(graphs_path+'/vmd_aliasing.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'/vmd_aliasing.tif',format='TIFF',dpi=600)

huaxian_dwt = pd.read_csv(root_path+"/Huaxian_dwt/data/db10-2/DWT_TRAIN.csv")
T=huaxian_dwt.shape[0]
t = np.arange(start=1,stop=T+1,step=1,dtype=np.float)/T
freqs = t-0.5-1/T
L = huaxian_dwt.shape[1]-1
columns=['D1','D2','A2',]
plt.figure(figsize=(7.48,7.48))
for i in range(1,L+1):
    plt.subplot(L,2,2*i-1)
    if i==L:
        plt.xlabel('Time(month)')
    plt.ylabel(columns[i-1])
    plt.plot(huaxian_dwt[columns[i-1]],color='b',label='',linewidth=0.8)

    plt.subplot(L,2,2*i)
    plt.plot(freqs,abs(fft(huaxian_dwt[columns[i-1]])),c='b',lw=0.8)
    if i==L:
        plt.xlabel('Frequence(1/month)')
    plt.ylabel('Amplitude')
plt.tight_layout()
# plt.savefig(graphs_path+'/vmd_aliasing.eps',format='EPS',dpi=2000)
# plt.savefig(graphs_path+'/vmd_aliasing.tif',format='TIFF',dpi=600)


Fs=huaxian_vmd.shape[0] #sampling frequency
T=1/Fs #sampling period(interval)
L=huaxian_vmd.shape[0]
t = np.arange(start=0,stop=L,step=1,dtype=np.float)*T #sampling times
plt.figure(figsize=(7.48,7.48))
plt.subplot(2,2,1)
Y_e = fft(huaxian_eemd['IMF1'])
P2_e=np.abs(Y_e/L)
P1_e=P2_e[1:int(L/2)+1]
P1_e[1:len(P1_e)-1]=2*P1_e[1:len(P1_e)-1]
f_e = Fs*np.arange(0,int(L/2),1)/L
plt.xlabel(r"f(Hz)")
plt.ylabel(r"P1\|f\|")
plt.plot(f_e,P1_e,c='b')

plt.subplot(2,2,2)
Y_s = fft(huaxian_ssa['Noise'])
P2_s=np.abs(Y_s/L)
P1_s=P2_s[1:int(L/2)+1]
P1_s[1:len(P1_s)-1]=2*P1_s[1:len(P1_s)-1]
f_s = Fs*np.arange(0,int(L/2),1)/L
plt.xlabel(r"f(Hz)")
plt.ylabel(r"P1\|f\|")
plt.plot(f_s,P1_s,c='b')

plt.subplot(2,2,3)
Y_v = fft(huaxian_vmd['IMF8'])
P2_v=np.abs(Y_v/L)
P1_v=P2_v[1:int(L/2)+1]
P1_v[1:len(P1_v)-1]=2*P1_v[1:len(P1_v)-1]
f_v = Fs*np.arange(0,int(L/2),1)/L
plt.xlabel(r"f(Hz)")
plt.ylabel(r"P1\|f\|")
plt.plot(f_v,P1_v,c='b')

plt.subplot(2,2,4)
Y_w = fft(huaxian_dwt['D1'])
P2_w=np.abs(Y_w/L)
P1_w=P2_w[1:int(L/2)+1]
P1_w[1:len(P1_w)-1]=2*P1_w[1:len(P1_w)-1]
f_w = Fs*np.arange(0,int(L/2),1)/L
plt.xlabel(r"f(Hz)")
plt.ylabel(r"P1\|f\|")
plt.plot(f_w,P1_w,c='b')


plt.figure(figsize=(3.54,2.0))
plt.subplot(2,2,1)
plt.text(-0.53,265,'(a)',fontsize=7)
# plt.xlabel('Frequence(1/month)')
plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(huaxian_eemd['IMF1'])),c='b',lw=0.8)

plt.subplot(2,2,2)
plt.text(-0.53,95,'(b)',fontsize=7)
# plt.xlabel('Frequence(1/month)')
# plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(huaxian_ssa['Periodic5'])),c='b',lw=0.8)

plt.subplot(2,2,3)
plt.text(-0.53,180,'(c)',fontsize=7)
plt.xlabel('Frequence(1/month)')
plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(huaxian_vmd['IMF8'])),c='b',lw=0.8)

plt.subplot(2,2,4)
plt.text(-0.53,220,'(d)',fontsize=7)
plt.xlabel('Frequence(1/month)')
# plt.ylabel('Amplitude')
plt.plot(freqs,abs(fft(huaxian_dwt['D1'])),c='b',lw=0.8)
plt.tight_layout()
plt.savefig(graphs_path+"frequency_spectrum.eps",format="EPS",dpi=2000)
plt.savefig(graphs_path+"frequency_spectrum.tif",format="TIFF",dpi=1200)
# plt.subplots_adjust(left=0.13, bottom=0.14, right=0.98,top=0.99, hspace=0.3, wspace=0.22)


#######################################################################
plt.figure(figsize=(3.54,3.54))
y_fqs_dom = fft(huaxian_eemd['IMF1'])
N=huaxian_eemd.shape[0]

plt.plot(abs(y_fqs_dom))


# plt.subplots_adjust(left=0.13, bottom=0.14, right=0.98,top=0.99, hspace=0.3, wspace=0.22)
plt.show()