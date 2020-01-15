import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.size']=8

import os
root_path = os.path.dirname(os.path.abspath('__file__'))
graph_path=root_path+'/graphs/'

def plot_pacfs(
    pacf_path,
    up_bound_path,
    low_bound_path,
    save_path,
    format='EPS',
    dpi=2000,
    ):
    """
    Plot partial autocorrelation function(coefficient)

    """
    pacfs = pd.read_csv(pacf_path,header=None)
    print(pacfs)
    up_bounds = pd.read_csv(up_bound_path,header=None)
    low_bounds = pd.read_csv(low_bound_path,header=None)
    plt.figure(figsize=(7.4861,7.4861))
    lags=list(range(0,pacfs.shape[0]))
    t=list(range(-1,pacfs.shape[0]))
    z_line=np.zeros(len(t))
    print(pacfs.shape[1]+1)
    for i in range(1,pacfs.shape[1]+1):
        # plt.subplot(3, 2, i)
        plt.subplot(math.ceil(pacfs.shape[1]/2.0), 2, i)
        plt.title(r'$IMF_{'+str(i)+'}$',loc='left',)
        plt.xlim(-0,20)
        plt.ylim(-1,1)
        plt.xticks([0,5,10,15,20],)
        plt.yticks()
        if i==pacfs.shape[1]-1 or i==pacfs.shape[1]:
            plt.xlabel('Lag(month)', )
        plt.ylabel('PACF', )
        plt.bar(lags,pacfs.iloc[:,i-1],color='b',width=0.8)
        plt.plot(lags,up_bounds.iloc[:,i-1], '--', color='r', label='',linewidth=0.8)
        plt.plot(lags,low_bounds.iloc[:,i-1], '--', color='r', label='',linewidth=0.8)
        plt.plot(t,z_line, '-', color='blue', label='',linewidth=0.5)
    # plt.subplots_adjust(left=0.09, bottom=0.06, right=0.98,top=0.96, hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, format=format, dpi=dpi)
    plt.show()

def plot_pacf(
    pacf_path,
    up_bound_path,
    low_bound_path,
    save_path,
    subsignal_id=1,
    format='EPS',
    dpi=2000,
    ):
    """
    Plot partial autocorrelation function(coefficient)

    """
    pacfs = pd.read_csv(pacf_path,header=None)
    print(pacfs)
    up_bounds = pd.read_csv(up_bound_path,header=None)
    low_bounds = pd.read_csv(low_bound_path,header=None)
    plt.figure(figsize=(3.54,1.6))
    lags=list(range(0,pacfs.shape[0]))
    t=list(range(-1,pacfs.shape[0]))
    z_line=np.zeros(len(t))
    # plt.title(r'PACF of $IMF_1}$',loc='left',)
    plt.xlim(-0,20)
    plt.ylim(-1,1)
    plt.xticks([0,5,10,15,20],)
    plt.yticks()
    plt.xlabel('Lag(month)')
    plt.ylabel('PACF')
    plt.bar(lags,pacfs.iloc[:,subsignal_id-1],color='b',width=0.8)
    plt.plot(lags,up_bounds.iloc[:,subsignal_id-1], '--', color='r', label='',linewidth=0.8)
    plt.plot(lags,low_bounds.iloc[:,subsignal_id-1], '--', color='r', label='',linewidth=0.8)
    plt.plot(t,z_line, '-', color='blue', label='',linewidth=0.5)
    # plt.subplots_adjust(left=0.09, bottom=0.06, right=0.98,top=0.96, hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, format=format, dpi=dpi)
    plt.show()


if __name__ == "__main__":
    # # plot pacf for Huaxian_eemd
    # plot_pacf(
    #     pacf_path=root_path+'/Huaxian_eemd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Huaxian_eemd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Huaxian_eemd/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/huaxian-eemd-pacf.eps'
    # )

    # plot pacf for huaxian-ssa
    # plot_pacf(
    #     pacf_path=root_path+'/Huaxian_ssa/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Huaxian_ssa/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Huaxian_ssa/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/huaxian-ssa-pacf.eps'
    # )

    # # plot pacf for huaxian-vmd
    # plot_pacf(
    #     pacf_path=root_path+'/Huaxian_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Huaxian_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Huaxian_vmd/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/huaxian-vmd-pacf.eps'
    # )

    # # plot pacf for huaxian-dwt
    plot_pacf(
        pacf_path=root_path+'/Huaxian_dwt/data/db10-2/pacfs20.csv',
        up_bound_path=root_path+'/Huaxian_dwt/data/db10-2/up_bounds20.csv',
        low_bound_path=root_path+'/Huaxian_dwt/data/db10-2/lo_bounds20.csv',
        save_path=root_path+'/graphs/huaxian-dwt(db10-2)-pacf.eps'
    )

    # # plot pacf for xianyang-eemd
    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_eemd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_eemd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_eemd/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/xianyang-eemd-pacf.eps'
    # )

    # # plot pacf for xianyang-ssa
    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_ssa/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_ssa/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_ssa/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/xianyang-ssa-pacf.eps'
    # )

    # # plot pacf for xianyang-vmd
    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_vmd/data/lo_bounds20.csv',
    #     save_path = root_path+'/graphs/xianyang-vmd-pacf.eps'
    # )

    # # plot pacf for xianyang-dwt
    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_dwt/data/db10-2/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_dwt/data/db10-2/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_dwt/data/db10-2/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/xianyang-dwt(db10-2)-pacf.eps'
    # )

    # # plot pacf for zhangjiashan-eemd
    # plot_pacf(
    #     pacf_path=root_path+'/Zhangjiashan_eemd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Zhangjiashan_eemd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Zhangjiashan_eemd/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/zhangjiashan-eemd-pacf.eps'
    # )

    # plot pacf for zhangjiashan-ssa
    plot_pacf(
        pacf_path=root_path+'/Zhangjiashan_ssa/data/pacfs20.csv',
        up_bound_path=root_path+'/Zhangjiashan_ssa/data/up_bounds20.csv',
        low_bound_path=root_path+'/Zhangjiashan_ssa/data/lo_bounds20.csv',
        save_path=root_path+'/graphs/zhangjiashan-ssa-pacf.eps'
    )

    # # plot pacf for zhangjiashan-vmd
    # plot_pacf(
    #     pacf_path=root_path+'/Zhangjiashan_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Zhangjiashan_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Zhangjiashan_vmd/data/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/Zhangjiashan_vmd-pacf.eps'
    # )

    # # plot pacf for zhangjiashan-dwt
    # plot_pacf(
    #     pacf_path=root_path+'/Zhangjiashan_dwt/data/db10-2/pacfs20.csv',
    #     up_bound_path=root_path+'/Zhangjiashan_dwt/data/db10-2/up_bounds20.csv',
    #     low_bound_path=root_path+'/Zhangjiashan_dwt/data/db10-2/lo_bounds20.csv',
    #     save_path=root_path+'/graphs/zhangjiashan-dwt(db10-2)-pacf.eps'
    # )


