import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
from datetime import datetime,timedelta
from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from statsmodels.tsa.stattools import adfuller
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    cri_dic={}
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
       cri_dic['Critical Value (%s)'%key]=value
    print (dfoutput)
    print(cri_dic)
    print(dfoutput['Test Statistic'])
    return cri_dic,dfoutput['Test Statistic']

def get_test_results(df,columns):
    test_statistic_list=[]
    for col in columns:
        timeseries = df[col]
        critical_dict,test_statistic=adf_test(timeseries)
        test_statistic_list.append(test_statistic)
    return critical_dict,test_statistic_list



eemd_train = pd.read_csv(root_path+"/Huaxian_eemd/data/EEMD_TRAIN.csv")
ssa_train = pd.read_csv(root_path+"/Huaxian_ssa/data/SSA_TRAIN.csv")
vmd_train = pd.read_csv(root_path+"/Huaxian_vmd/data/VMD_TRAIN.csv")
dwt_train = pd.read_csv(root_path+"/Huaxian_wd/data/db10-lev2/WD_TRAIN.csv")

c_eemd_hua,t_list_eemd_hua=get_test_results(df=eemd_train,columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9'])
c_ssa_hua,t_list_ssa_hua=get_test_results(df=ssa_train,columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'])
c_vmd_hua,t_list_vmd_hua=get_test_results(df=vmd_train,columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8'])
c_dwt_hua,t_list_dwt_hua=get_test_results(df=dwt_train,columns=['D1','D2','A2'])

t_lists=[
    t_list_eemd_hua,
    t_list_ssa_hua,
    t_list_vmd_hua,
    t_list_dwt_hua,
]

labels=["EEMD","SSA","VMD","DWT"]
markers=["v","d","o","+"]

plt.figure(figsize=(3.54,3.4))
for key,value in c_vmd_hua.items():
    plt.axhline(value,label=key,linestyle='--')
for i in range(len(t_lists)):
    plt.scatter(list(range(1,len(t_lists[i])+1)),t_lists[i],label=labels[i],marker=markers[i])
# ax=plt.gca()
# ax.set_xticklabels([r"$S_1$",r"$S_2$",r"$S_3$",r"$S_4$",r"$S_5$",r"$S_6$",r"$S_7$",r"$S_8$",r"$S_9$",r"$S_{10}$",r"$S_{11}$",r"$S_{12}$",])

    
plt.legend()
plt.show()
    



# adf_test(vmd_train["IMF8"])
# adf_test(eemd_train["IMF1"])

# plt.figure()
# plt.plot(vmd_train["IMF8"],label='VMD:IMF8',zorder=1)
# plt.plot(eemd_train["IMF1"],label="EEMD:IMF1",zorder=0)
# plt.legend()

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(dwt_train['D1'])
# plt.subplot(3,1,2)
# plt.plot(dwt_train['D2'])
# plt.subplot(3,1,3)
# plt.plot(dwt_train['A2'])

# plt.show()
