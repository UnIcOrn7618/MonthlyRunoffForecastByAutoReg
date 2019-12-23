#%%
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
parent_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = root_path + '\\Zhangjiashan_ssa\\data\\'
print(10 * '-' + ' Current Path: {}'.format(root_path))
print(10 * '-' + ' Parent Path: {}'.format(parent_path)) 
print(10 * '-' + ' Grandpa Path: {}'.format(grandpa_path)) 
print(10 * '-' + ' Data Path: {}'.format(data_path)) 

import sys
sys.path.append(root_path+'/tools/')
from ssa import SSA
# Loading the monthly runoff of Zhangjiashan station
Zhangjiashan = pd.read_excel(root_path+'/time_series/ZhangJiaShanRunoff1953-2018(1953-2018).xlsx')
Zhangjiashan = Zhangjiashan['MonthlyRunoff']

# plotting monthly Runoff of Zhangjiashan
Zhangjiashan.plot()
plt.title("Monthly Runoff of Zhangjiashan station")
plt.xlabel("Time(1953/01-2008/12)")
plt.ylabel(r"Runoff($m^3/s$)")
plt.tight_layout()

start = 0
stop = 672#552
full = Zhangjiashan[start:] #(full)from 1953/01 to 2018/12 792 samples
full = full.reset_index(drop=True)
train = Zhangjiashan[start:stop] #(train)from 1953/01 to 1998/12, 552 samples
train = train.reset_index(drop=True)


#%%
# Decompose the monthly runoff of Zhangjiashan
window = 12 #45% of 552
Zhangjiashan_ssa = SSA(full,window)
F0 = Zhangjiashan_ssa.reconstruct(0)
F1 = Zhangjiashan_ssa.reconstruct(1)
F2 = Zhangjiashan_ssa.reconstruct(2)
F3 = Zhangjiashan_ssa.reconstruct(3)
F4 = Zhangjiashan_ssa.reconstruct(4)
F5 = Zhangjiashan_ssa.reconstruct(5)
F6 = Zhangjiashan_ssa.reconstruct(6)
F7 = Zhangjiashan_ssa.reconstruct(7)
F8 = Zhangjiashan_ssa.reconstruct(8)
F9 = Zhangjiashan_ssa.reconstruct(9)
F10 = Zhangjiashan_ssa.reconstruct(10)
F11 = Zhangjiashan_ssa.reconstruct(11)
orig_TS = Zhangjiashan_ssa.orig_TS
df = pd.concat([F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,orig_TS],axis=1)
df = pd.DataFrame(df.values,columns=[
    'Trend',#F0
    'Periodic1',#F1
    'Periodic2',#F2
    'Periodic3',#F3
    'Periodic4',#F4
    'Periodic5',#F5
    'Periodic6',#F6
    'Periodic7',#F7
    'Periodic8',#F8
    'Periodic9',#F9
    'Periodic10',#F10
    'Noise',#F11
    'ORIG'#orig_TS
    ])
df.to_csv(data_path+'SSA_FULL.csv',index=None)
df

#%%
Zhangjiashan_ssa = SSA(train,window)
F0 = Zhangjiashan_ssa.reconstruct(0)
F1 = Zhangjiashan_ssa.reconstruct(1)
F2 = Zhangjiashan_ssa.reconstruct(2)
F3 = Zhangjiashan_ssa.reconstruct(3)
F4 = Zhangjiashan_ssa.reconstruct(4)
F5 = Zhangjiashan_ssa.reconstruct(5)
F6 = Zhangjiashan_ssa.reconstruct(6)
F7 = Zhangjiashan_ssa.reconstruct(7)
F8 = Zhangjiashan_ssa.reconstruct(8)
F9 = Zhangjiashan_ssa.reconstruct(9)
F10 = Zhangjiashan_ssa.reconstruct(10)
F11 = Zhangjiashan_ssa.reconstruct(11)
orig_TS = Zhangjiashan_ssa.orig_TS
df = pd.concat([F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,orig_TS],axis=1)
df = pd.DataFrame(df.values,columns=[
    'Trend',#F0
    'Periodic1',#F1
    'Periodic2',#F2
    'Periodic3',#F3
    'Periodic4',#F4
    'Periodic5',#F5
    'Periodic6',#F6
    'Periodic7',#F7
    'Periodic8',#F8
    'Periodic9',#F9
    'Periodic10',#F10
    'Noise',#F11
    'ORIG'#orig_TS
    ])
df.to_csv(data_path+'SSA_TRAIN672.csv',index=None)

#%%
# if not os.path.exists(data_path+'ssa-test'):
#     os.makedirs(data_path+'ssa-test')
# for i in range(1,241):
#     data = Zhangjiashan[start:stop+i]
#     data = data.reset_index(drop=True)
#     Zhangjiashan_ssa = SSA(data,window)
#     F0 = Zhangjiashan_ssa.reconstruct(0)
#     F1 = Zhangjiashan_ssa.reconstruct(1)
#     F2 = Zhangjiashan_ssa.reconstruct(2)
#     F3 = Zhangjiashan_ssa.reconstruct(3)
#     F4 = Zhangjiashan_ssa.reconstruct(4)
#     F5 = Zhangjiashan_ssa.reconstruct(5)
#     F6 = Zhangjiashan_ssa.reconstruct(6)
#     F7 = Zhangjiashan_ssa.reconstruct(7)
#     F8 = Zhangjiashan_ssa.reconstruct(8)
#     F9 = Zhangjiashan_ssa.reconstruct(9)
#     F10 = Zhangjiashan_ssa.reconstruct(10)
#     F11 = Zhangjiashan_ssa.reconstruct(11)
#     orig_TS = Zhangjiashan_ssa.orig_TS
#     df = pd.concat([F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,orig_TS],axis=1)
#     df = pd.DataFrame(df.values,columns=[
#         'Trend',#F0
#         'Periodic1',#F1
#         'Periodic2',#F2
#         'Periodic3',#F3
#         'Periodic4',#F4
#         'Periodic5',#F5
#         'Periodic6',#F6
#         'Periodic7',#F7
#         'Periodic8',#F8
#         'Periodic9',#F9
#         'Periodic10',#F10
#         'Noise',#F11
#         'ORIG'#orig_TS
#         ])
#     df.to_csv(data_path+'ssa-test/ssa_appended_test'+str(552+i)+'.csv',index=None)

#%%
