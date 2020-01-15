import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
from skopt import dump, load
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir)) # For run in CMD
graphs_path = root_path+'/results_analysis/graphs/'

import sys
sys.path.append(root_path)
from skopt_plots import plot_convergence,plot_objective,plot_evaluations

res = load(root_path+'/Huaxian_vmd/projects/esvr/one_step_1_month_forecast/result_seed4.pkl')
fig = plt.figure(num=1,figsize=(3.54,2.51))
ax0 = fig.add_subplot(111)
plot_convergence(res,ax=ax0, true_minimum=0.0,)
plt.title("")
ax0.set_ylabel(r"Minimum $MSE$ after $n$ calls")
plt.tight_layout()
plt.savefig(graphs_path+'convergence_huaxian_vmd_seed4.eps',format="EPS",dpi=2000)
plt.savefig(graphs_path+'convergence_huaxian_vmd_seed4.tif',format="TIFF",dpi=1200)

plot_objective(res,figsize=(4.5,4.5),dimensions=[r'$C$',r'$\epsilon$',r'$\sigma$'])
# plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
plt.tight_layout()
plt.savefig(graphs_path+'objective_huaxian_vmd_seed4.eps',format="EPS",dpi=2000)
plt.savefig(graphs_path+'objective_huaxian_vmd_seed4.tif',format="TIFF",dpi=1200)
plt.show()