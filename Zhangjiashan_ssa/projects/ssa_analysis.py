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
plt.figure()
Zhangjiashan.plot()
plt.title("Monthly Runoff of Zhangjiashan station")
plt.xlabel("Time(1953/01-2008/12)")
plt.ylabel(r"Runoff($m^3/s$)")
plt.tight_layout()

# loading the training and entire dataset
start = 0
stop = 552
data = Zhangjiashan[start:stop] # (train) from 1953/01 to 1998/12, 552 samples
# data = Zhangjiashan[start:] # (full) from 1953/01 to 1998/12, 552 samples
data = data.reset_index(drop=True)

Zhangjiashan


#%%
# Decomposing the monthly Runoff of Zhangjiashan With SSA
window = 248
Zhangjiashan_ssa = SSA(Zhangjiashan,window)
plt.figure()
Zhangjiashan_ssa.plot_wcorr()
plt.title("W-Correlation for monthly Runoff of Zhangjiashan")
plt.title("Monthly Runoff of Zhangjiashan station")
plt.xlabel("Time(1953/01-1998/12)")
plt.ylabel(r"Runoff($m^3/s$)")
plt.tight_layout()

#%%
# Of course, with a larger windown length (and therefore a large number
# of elementary components), such a view of the w-correlation matrix is 
# not the most helpful. Zoom into the w-correlation matrix for the first
# 50 components
plt.figure()
Zhangjiashan_ssa.plot_wcorr(max=59)
plt.title("W-Correlation for the monthly Runoff of Zhangjiashan")


#%%
plt.figure()
Zhangjiashan_ssa.reconstruct(0).plot()
Zhangjiashan_ssa.reconstruct([1,2]).plot()
Zhangjiashan_ssa.reconstruct([3,4]).plot()
Zhangjiashan_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Zhangjiashan: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
Zhangjiashan_ssa.reconstruct(slice(0,5)).plot()
Zhangjiashan_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Zhangjiashan: Low-Frequancy Periodicity")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
# plt.xlim(16,20)
# plt.ylim(0,10)



#%%
plt.figure()
Zhangjiashan_ssa.reconstruct(0).plot()
Zhangjiashan_ssa.reconstruct([1,2]).plot()
Zhangjiashan_ssa.reconstruct([3,4]).plot()
Zhangjiashan_ssa.reconstruct(slice(5,window)).plot()
Zhangjiashan_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Zhangjiashan: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
Zhangjiashan_ssa.reconstruct(0).plot()
Zhangjiashan_ssa.reconstruct([1,2]).plot()
Zhangjiashan_ssa.reconstruct([3,4]).plot()
Zhangjiashan_ssa.reconstruct(slice(5,11)).plot()
Zhangjiashan_ssa.reconstruct(slice(11,window)).plot()
Zhangjiashan_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Zhangjiashan: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(5)]+["Original TS"]
plt.legend(legend)


#%%
# The final decomposition
plt.figure()
Zhangjiashan_ssa.reconstruct(0).plot()
Zhangjiashan_ssa.reconstruct([1,2]).plot()
Zhangjiashan_ssa.reconstruct([3,4]).plot()
Zhangjiashan_ssa.reconstruct([5,6]).plot()
Zhangjiashan_ssa.reconstruct([7,8]).plot()
Zhangjiashan_ssa.reconstruct(slice(9,13)).plot()
Zhangjiashan_ssa.reconstruct([13,14]).plot()
Zhangjiashan_ssa.reconstruct(slice(15,window)).plot()
Zhangjiashan_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Zhangjiashan: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(8)]+["Original TS"]
plt.legend(legend)
plt.show()

#%%
