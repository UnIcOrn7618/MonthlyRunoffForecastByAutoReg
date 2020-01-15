#%%
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__'))

import sys
sys.path.append(root_path+'/config/')
from variables import train_len, dev_len,test_len
from ssa import SSA



station = 'Huaxian' # 'Huaxian', 'Xianyang' and 'Zhangjiashan'
save_path = {
    'Huaxian':root_path + '\\Huaxian_ssa\\data\\',
    'Xianyang':root_path + '\\Xianyang_ssa\\data\\',
    'Zhangjiashan':root_path + '\\Zhangjiashan_ssa\\data\\',
}
data ={
    'Huaxian':(pd.read_excel(root_path+'/time_series/HuaxianRunoff1951-2018(1953-2018).xlsx')['MonthlyRunoff'][24:]).reset_index(drop=True),
    'Xianyang':(pd.read_excel(root_path+'/time_series/XianyangRunoff1951-2018(1953-2018).xlsx')['MonthlyRunoff'][24:]).reset_index(drop=True),
    'Zhangjiashan':(pd.read_excel(root_path+'/time_series/ZhangJiaShanRunoff1953-2018(1953-2018).xlsx')['MonthlyRunoff'][0:]).reset_index(drop=True),
}

if not os.path.exists(save_path[station]+'ssa-test\\'):
    os.makedirs(save_path[station]+'ssa-test\\')



# plotting the monthly runoff of huaxian station
data[station].plot()
plt.title("Monthly Runoff of "+station+" station")
plt.xlabel("Time(1953/01-2008/12)")
plt.ylabel(r"Runoff($m^3/s$)")
plt.tight_layout()
plt.show()

full = data[station] #(full)from 1953/01 to 2018/12 792 samples
train = full[:train_len] #(train)from 1953/01 to 1998/12, 552 samples
train_dev = full[:train_len+dev_len]

# decomposition parameter
window = 12
columns=[
    'ORIG',#orig_TS
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
]

#%%
# Decompose the entire monthly runoff of huaxian
huaxian_ssa = SSA(full,window)
F0 = huaxian_ssa.reconstruct(0)
F1 = huaxian_ssa.reconstruct(1)
F2 = huaxian_ssa.reconstruct(2)
F3 = huaxian_ssa.reconstruct(3)
F4 = huaxian_ssa.reconstruct(4)
F5 = huaxian_ssa.reconstruct(5)
F6 = huaxian_ssa.reconstruct(6)
F7 = huaxian_ssa.reconstruct(7)
F8 = huaxian_ssa.reconstruct(8)
F9 = huaxian_ssa.reconstruct(9)
F10 = huaxian_ssa.reconstruct(10)
F11 = huaxian_ssa.reconstruct(11)
orig_TS = huaxian_ssa.orig_TS
df = pd.concat([orig_TS,F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11],axis=1)
df = pd.DataFrame(df.values,columns=columns)
df.to_csv(save_path[station]+'SSA_FULL.csv',index=None)

#%%
# Decompose the training monthly runoff of huaxian
huaxian_ssa = SSA(train,window)
F0 = huaxian_ssa.reconstruct(0)
F1 = huaxian_ssa.reconstruct(1)
F2 = huaxian_ssa.reconstruct(2)
F3 = huaxian_ssa.reconstruct(3)
F4 = huaxian_ssa.reconstruct(4)
F5 = huaxian_ssa.reconstruct(5)
F6 = huaxian_ssa.reconstruct(6)
F7 = huaxian_ssa.reconstruct(7)
F8 = huaxian_ssa.reconstruct(8)
F9 = huaxian_ssa.reconstruct(9)
F10 = huaxian_ssa.reconstruct(10)
F11 = huaxian_ssa.reconstruct(11)
orig_TS = huaxian_ssa.orig_TS
df = pd.concat([orig_TS,F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11],axis=1)
df = pd.DataFrame(df.values,columns=columns)
df.to_csv(save_path[station]+'SSA_TRAIN.csv',index=None)

#%%
# Decompose the training-development monthly runoff of huaxian
huaxian_ssa = SSA(train_dev,window)
F0 = huaxian_ssa.reconstruct(0)
F1 = huaxian_ssa.reconstruct(1)
F2 = huaxian_ssa.reconstruct(2)
F3 = huaxian_ssa.reconstruct(3)
F4 = huaxian_ssa.reconstruct(4)
F5 = huaxian_ssa.reconstruct(5)
F6 = huaxian_ssa.reconstruct(6)
F7 = huaxian_ssa.reconstruct(7)
F8 = huaxian_ssa.reconstruct(8)
F9 = huaxian_ssa.reconstruct(9)
F10 = huaxian_ssa.reconstruct(10)
F11 = huaxian_ssa.reconstruct(11)
orig_TS = huaxian_ssa.orig_TS
df = pd.concat([orig_TS,F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,],axis=1)
df = pd.DataFrame(df.values,columns=columns)
df.to_csv(save_path[station]+'SSA_TRAINDEV.csv',index=None)

#%%
for i in range(1,241):
    data = full[0:train_len+i]
    huaxian_ssa = SSA(data,window)
    F0 = huaxian_ssa.reconstruct(0)
    F1 = huaxian_ssa.reconstruct(1)
    F2 = huaxian_ssa.reconstruct(2)
    F3 = huaxian_ssa.reconstruct(3)
    F4 = huaxian_ssa.reconstruct(4)
    F5 = huaxian_ssa.reconstruct(5)
    F6 = huaxian_ssa.reconstruct(6)
    F7 = huaxian_ssa.reconstruct(7)
    F8 = huaxian_ssa.reconstruct(8)
    F9 = huaxian_ssa.reconstruct(9)
    F10 = huaxian_ssa.reconstruct(10)
    F11 = huaxian_ssa.reconstruct(11)
    orig_TS = huaxian_ssa.orig_TS
    df = pd.concat([orig_TS,F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,],axis=1)
    df = pd.DataFrame(df.values,columns=columns)
    df.to_csv(save_path[station]+'ssa-test/ssa_appended_test'+str(train_len+i)+'.csv',index=None)
