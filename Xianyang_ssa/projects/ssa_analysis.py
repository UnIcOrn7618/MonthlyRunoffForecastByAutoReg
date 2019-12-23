#%%
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
parent_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = root_path + '\\Xianyang_ssa\\data\\'
print(10 * '-' + ' Current Path: {}'.format(root_path))
print(10 * '-' + ' Parent Path: {}'.format(parent_path)) 
print(10 * '-' + ' Grandpa Path: {}'.format(grandpa_path)) 
print(10 * '-' + ' Data Path: {}'.format(data_path)) 

import sys
sys.path.append(root_path+'/tools/')
from ssa import SSA
# Loading the monthly runoff of xianyang station
xianyang = pd.read_excel(root_path+'/time-series/XianyangRunoff1951-2018(1953-2018).xlsx')
xianyang = xianyang['MonthlyRunoff'][24:576] #from 1953/01 to 1998/12, 552 samples
# plotting the data
plt.figure()
xianyang.plot()
xianyang


#%%
# Decomposing the monthly Runoff of xianyang With SSA
window = 248
xianyang_ssa = SSA(xianyang,window)
plt.figure()
xianyang_ssa.plot_wcorr()
plt.title("W-Correlation for monthly Runoff of xianyang")
plt.tight_layout()


#%%
# Of course, with a larger windown length (and therefore a large number
# of elementary components), such a view of the w-correlation matrix is 
# not the most helpful. Zoom into the w-correlation matrix for the first
# 50 components
plt.figure()
xianyang_ssa.plot_wcorr(max=49)
plt.title("W-Correlation for the monthly Runoff of xianyang")


#%%
plt.figure()
xianyang_ssa.reconstruct(0).plot()
xianyang_ssa.reconstruct([1,2]).plot()
xianyang_ssa.reconstruct([3,4]).plot()
xianyang_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of xianyang: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
xianyang_ssa.reconstruct(slice(0,5)).plot()
xianyang_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of xianyang: Low-Frequancy Periodicity")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
# plt.xlim(16,20)
# plt.ylim(0,10)



#%%
plt.figure()
xianyang_ssa.reconstruct(0).plot()
xianyang_ssa.reconstruct([1,2]).plot()
xianyang_ssa.reconstruct([3,4]).plot()
xianyang_ssa.reconstruct(slice(5,window)).plot()
xianyang_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of xianyang: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
xianyang_ssa.reconstruct(0).plot()
xianyang_ssa.reconstruct([1,2]).plot()
xianyang_ssa.reconstruct([3,4]).plot()
xianyang_ssa.reconstruct(slice(5,11)).plot()
xianyang_ssa.reconstruct(slice(11,window)).plot()
xianyang_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of xianyang: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(5)]+["Original TS"]
plt.legend(legend)


#%%
# The final decomposition
plt.figure()
xianyang_ssa.reconstruct(0).plot()
xianyang_ssa.reconstruct([1,2]).plot()
xianyang_ssa.reconstruct([3,4]).plot()
xianyang_ssa.reconstruct(slice(5,11)).plot()
xianyang_ssa.reconstruct([11,12]).plot()
xianyang_ssa.reconstruct([14]).plot()
xianyang_ssa.reconstruct([15,16]).plot()
xianyang_ssa.reconstruct([13]+[i for i in range(17,window)]).plot()
# xianyang_ssa.reconstruct([24,25]).plot()
# xianyang_ssa.reconstruct(slice(26,210)).plot()
xianyang_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of xianyang: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(8)]+["Original TS"]
plt.legend(legend)
plt.show()

#%%
