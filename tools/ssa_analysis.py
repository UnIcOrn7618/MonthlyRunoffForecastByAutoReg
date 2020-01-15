#%%
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
parent_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = parent_path + '\\data\\'
print(10 * '-' + ' Current Path: {}'.format(root_path))
print(10 * '-' + ' Parent Path: {}'.format(parent_path)) 
print(10 * '-' + ' Grandpa Path: {}'.format(grandpa_path)) 
print(10 * '-' + ' Data Path: {}'.format(data_path)) 

import sys
root_path=grandpa_path
sys.path.append(root_path)
from ssa import SSA
# Loading the monthly runoff of Huaxian station
huaxian = pd.read_excel(root_path+'/time_series/HuaxianRunoff1951-2018(1953-2018).xlsx')
huaxian = huaxian['MonthlyRunoff'][24:576] #from 1953/01 to 1998/12, 552 samples
# plotting the data
plt.figure()
huaxian.plot()
plt.xlabel("Time(1953/01-1998/12)")
plt.ylabel(r"Runoff($m^3/s$)")

#%%
# Decomposing the monthly Runoff of huaxian With SSA
window = 12
huaxian_ssa = SSA(huaxian,window)
plt.figure()
huaxian_ssa.plot_wcorr()
plt.title("W-Correlation for monthly Runoff of Huaxian")
plt.tight_layout()


#%%
# Of course, with a larger windown length (and therefore a large number
# of elementary components), such a view of the w-correlation matrix is 
# not the most helpful. Zoom into the w-correlation matrix for the first
# 50 components
print("corr:\n{}".format(huaxian_ssa.calc_wcorr()))
plt.figure(figsize=(5.51,5))
huaxian_ssa.plot_wcorr(max=11)
plt.title("W-Correlation for the monthly Runoff of Huaxian",fontsize=10)
plt.subplots_adjust(left=0.12, bottom=0.06, right=0.9,top=0.98, hspace=0.4, wspace=0.25)
# plt.savefig(root_path+'/Huaxian_ssa/graphs/w_correlation.eps',format='EPS',dpi=2000)
# plt.savefig(root_path+'/Huaxian_ssa/graphs/w_correlation.tif',format='TIFF',dpi=1000)
plt.show()
print("@@@@")

#%%
plt.figure()
huaxian_ssa.reconstruct(0).plot()
huaxian_ssa.reconstruct([1,2]).plot()
huaxian_ssa.reconstruct([3,4]).plot()
huaxian_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Huaxian: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(3)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
huaxian_ssa.reconstruct(slice(0,5)).plot()
huaxian_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Huaxian: Low-Frequancy Periodicity")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
# plt.xlim(16,20)
# plt.ylim(0,10)



#%%
plt.figure()
huaxian_ssa.reconstruct(0).plot()
huaxian_ssa.reconstruct([1,2]).plot()
huaxian_ssa.reconstruct([3,4]).plot()
huaxian_ssa.reconstruct(slice(5,window)).plot()
huaxian_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Huaxian: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)]+["Original TS"]
plt.legend(legend)




#%%
plt.figure()
huaxian_ssa.reconstruct(0).plot()
huaxian_ssa.reconstruct([1,2]).plot()
huaxian_ssa.reconstruct([3,4]).plot()
huaxian_ssa.reconstruct(slice(5,11)).plot()
huaxian_ssa.reconstruct(slice(11,window)).plot()
huaxian_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Huaxian: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(5)]+["Original TS"]
plt.legend(legend)


#%%
# final decomposition level
plt.figure()
plt.subplot(6,2,1)
huaxian_ssa.reconstruct(0).plot()
plt.subplot(6,2,2)
huaxian_ssa.reconstruct(1).plot()
plt.subplot(6,2,3)
huaxian_ssa.reconstruct(2).plot()
plt.subplot(6,2,4)
huaxian_ssa.reconstruct(3).plot()
plt.subplot(6,2,5)
huaxian_ssa.reconstruct(4).plot()
plt.subplot(6,2,6)
huaxian_ssa.reconstruct(5).plot()
plt.subplot(6,2,7)
huaxian_ssa.reconstruct(6).plot()
plt.subplot(6,2,8)
huaxian_ssa.reconstruct(7).plot()
plt.subplot(6,2,9)
huaxian_ssa.reconstruct(8).plot()
plt.subplot(6,2,10)
huaxian_ssa.reconstruct(9).plot()
plt.subplot(6,2,11)
huaxian_ssa.reconstruct(10).plot()
plt.subplot(6,2,12)
huaxian_ssa.reconstruct(11).plot()
# huaxian_ssa.orig_TS.plot(alpha=0.4)
plt.title("Monthly Runoff of Huaxian: First Three groups")
plt.xlabel(r"$t$(month)")
plt.ylabel(r"Runoff($m^3/s$)")
legend=[r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(12)]+["Original TS"]
plt.legend(legend)
plt.show()