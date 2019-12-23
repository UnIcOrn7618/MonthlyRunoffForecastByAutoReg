import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'/results_analysis/graphs/'

import sys
sys.path.append(root_path+'/tools/')
from plot_pacfs import plot_pacf,plot_pacfs

if __name__ == "__main__":
    # plot_pacf(
    #     pacf_path=root_path+'/Huaxian_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Huaxian_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Huaxian_vmd/data/lo_bounds20.csv',
    #     save_path=graphs_path+'/huaxian-vmd-pacf.eps',
    # )
    plot_pacfs(
        pacf_path=root_path+'/Huaxian_vmd/data/pacfs20.csv',
        up_bound_path=root_path+'/Huaxian_vmd/data/up_bounds20.csv',
        low_bound_path=root_path+'/Huaxian_vmd/data/lo_bounds20.csv',
        save_path=graphs_path+'/huaxian-vmd-pacf.tif',
        format='TIFF',
        dpi=600,
    )
    plot_pacf(
        pacf_path=root_path+'/Huaxian_vmd/data/pacfs20.csv',
        up_bound_path=root_path+'/Huaxian_vmd/data/up_bounds20.csv',
        low_bound_path=root_path+'/Huaxian_vmd/data/lo_bounds20.csv',
        save_path=graphs_path+'/huaxian-vmd-imf1-pacf.tif',
        subsignal_id=1,
        format='TIFF',
        dpi=1200,
    )

    


    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_vmd/data/lo_bounds20.csv',
    #     save_path = graphs_path+'/xianyang-vmd-pacf.eps'
    # )
    # plot_pacf(
    #     pacf_path=root_path+'/Xianyang_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Xianyang_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Xianyang_vmd/data/lo_bounds20.csv',
    #     save_path = graphs_path+'/xianyang-vmd-pacf.tif',
    #     format='TIFF',
    #     dpi=1000,
    # )

    # plot_pacf(
    #     pacf_path=root_path+'/Zhangjiashan_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Zhangjiashan_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Zhangjiashan_vmd/data/lo_bounds20.csv',
    #     save_path=graphs_path+'/Zhangjiashan_vmd-pacf.eps'
    # )
    # plot_pacf(
    #     pacf_path=root_path+'/Zhangjiashan_vmd/data/pacfs20.csv',
    #     up_bound_path=root_path+'/Zhangjiashan_vmd/data/up_bounds20.csv',
    #     low_bound_path=root_path+'/Zhangjiashan_vmd/data/lo_bounds20.csv',
    #     save_path=graphs_path+'/Zhangjiashan_vmd-pacf.tif',
    #     format='TIFF',
    #     dpi=1000,
    # )

    