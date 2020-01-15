import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
import math
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graphs_path = root_path+'\\results_analysis\\graphs\\'

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import sys
sys.path.append(root_path)
from metrics_ import PPTS,mean_absolute_percentage_error
from results_reader import read_pca_metrics



if __name__ == "__main__":
    # read metrics of vmd-svr models for all three stations
    h_v_mle,h_v_r2,h_v_nrmse,h_v_mae,h_v_mape,h_v_ppts=read_pca_metrics(station="Huaxian",decomposer="vmd",start_component=15,stop_component=30)
    x_v_mle,x_v_r2,x_v_nrmse,x_v_mae,x_v_mape,x_v_ppts=read_pca_metrics(station="Xianyang",decomposer="vmd",start_component=15,stop_component=30)
    z_v_mle,z_v_r2,z_v_nrmse,z_v_mae,z_v_mape,z_v_ppts=read_pca_metrics(station="Zhangjiashan",decomposer="vmd",start_component=11,stop_component=26)

    # read metrics of eemd-svr models for all three stations
    h_e_mle,h_e_r2,h_e_nrmse,h_e_mae,h_e_mape,h_e_ppts=read_pca_metrics(station="Huaxian",decomposer="eemd",start_component=36,stop_component=51)
    x_e_mle,x_e_r2,x_e_nrmse,x_e_mae,x_e_mape,x_e_ppts=read_pca_metrics(station="Xianyang",decomposer="eemd",start_component=32,stop_component=47)
    z_e_mle,z_e_r2,z_e_nrmse,z_e_mae,z_e_mape,z_e_ppts=read_pca_metrics(station="Zhangjiashan",decomposer="eemd",start_component=39,stop_component=54)

    # read metrics of ssa-svr models for all three stations
    h_s_mle,h_s_r2,h_s_nrmse,h_s_mae,h_s_mape,h_s_ppts=read_pca_metrics(station="Huaxian",decomposer="ssa",start_component=40,stop_component=55)
    x_s_mle,x_s_r2,x_s_nrmse,x_s_mae,x_s_mape,x_s_ppts=read_pca_metrics(station="Xianyang",decomposer="ssa",start_component=39,stop_component=54)
    z_s_mle,z_s_r2,z_s_nrmse,z_s_mae,z_s_mape,z_s_ppts=read_pca_metrics(station="Zhangjiashan",decomposer="ssa",start_component=35,stop_component=50)

    # read metrics of dwt-svr models for all three stations
    h_w_mle,h_w_r2,h_w_nrmse,h_w_mae,h_w_mape,h_w_ppts=read_pca_metrics(station="Huaxian",decomposer="dwt",start_component=45,stop_component=60)
    x_w_mle,x_w_r2,x_w_nrmse,x_w_mae,x_w_mape,x_w_ppts=read_pca_metrics(station="Xianyang",decomposer="dwt",start_component=45,stop_component=60)
    z_w_mle,z_w_r2,z_w_nrmse,z_w_mae,z_w_mape,z_w_ppts=read_pca_metrics(station="Zhangjiashan",decomposer="dwt",start_component=45,stop_component=60)

    print("Huaxian:VMD-SVR:{}\n".format((max(h_v_r2[1:])-h_v_r2[0])/h_v_r2[0]))
    print("Xianyang:VMD-SVR:{}\n".format((max(x_v_r2[1:])-x_v_r2[0])/x_v_r2[0]))
    print("Zhangjiashan:VMD-SVR:{}\n".format((max(z_v_r2[1:])-z_v_r2[0])/z_v_r2[0]))
    print("Huaxian:DWT-SVR:{}\n".format((max(h_w_r2[1:])-h_w_r2[0])/h_w_r2[0]))
    print("Xianyang:DWT-SVR:{}\n".format((max(x_w_r2[1:])-x_w_r2[0])/x_w_r2[0]))
    print("Zhangjiashan:DWT-SVR:{}\n".format((max(z_w_r2[1:])-z_w_r2[0])/z_w_r2[0]))
    

    plt.figure(figsize=(5.51,3))
    t=list(range(0,16))
    
    ax1=plt.subplot(2,2,1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.xlabel('Number of reduced predictors')
    plt.ylabel(r'$NSE$')
    plt.ylim(0.2,0.7)
    # plt.title("EEMD-SVR",loc="left",)
    plt.text(14.38,0.65,"(a)",fontweight='normal',fontsize=7)
    plt.xticks(np.arange(0,16,1))
    plt.plot(t,h_e_r2[1:],marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_e_r2[1:],marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_e_r2[1:],marker='v',label="Zhangjiashan",markerfacecolor='w')
    plt.axhline(h_e_r2[0],color='tab:blue',label='Huaxian:without PCA',linestyle='-',linewidth=1.5)
    plt.axhline(x_e_r2[0],color='tab:orange',label='Xianyang:without PCA',linestyle='-',linewidth=1.0)
    plt.axhline(z_e_r2[0],color='tab:green',label='Zhangjiashan:without PCA',linestyle='-',linewidth=0.5)
    plt.axvline(h_e_mle,color='tab:blue',label='Huaxian:PCA MLE',linestyle='--',linewidth=1.5)
    plt.axvline(x_e_mle,color='tab:orange',label='Xianyang:PCA MLE',linestyle='--',linewidth=1.0)
    plt.axvline(z_e_mle,color='tab:green',label='Zhangjiashan:PCA MLE',linestyle='--',linewidth=0.5)
    plt.legend(
        loc='upper left',
        # loc=0,
        # bbox_to_anchor=(0.08,1.01, 1,0.101),
        bbox_to_anchor=(0.25,1.55),
        # bbox_transform=plt.gcf().transFigure,
        ncol=3,
        shadow=False,
        frameon=True,
    )

    ax2=plt.subplot(2,2,2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.xlabel('Number of reduced predictors')
    # plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("SSA-SVR",loc="left",)
    plt.text(14.38,0.96,"(b)",fontweight='normal',fontsize=7)
    plt.xticks(np.arange(0,16,1))
    plt.plot(t,h_s_r2[1:],marker='o',label="Huaxian:SSA-SVR",markerfacecolor='w')
    plt.plot(t,x_s_r2[1:],marker='s',label="Xianyang:SSA-SVR",markerfacecolor='w')
    plt.plot(t,z_s_r2[1:],marker='v',label="Zhangjiashan:SSA-SVR",markerfacecolor='w')
    plt.axhline(h_s_r2[0],color='tab:blue',label='Huaxian:without PCA',linestyle='-',linewidth=1.5)
    plt.axhline(x_s_r2[0],color='tab:orange',label='Xianyang:without PCA',linestyle='-',linewidth=1.0)
    plt.axhline(z_s_r2[0],color='tab:green',label='Zhangjiashan:without PCA',linestyle='-',linewidth=0.5)
    plt.axvline(h_s_mle,color='tab:blue',label='Huaxian:PCA MLE',linestyle='--',linewidth=1.5)
    plt.axvline(x_s_mle,color='tab:orange',label='Xianyang:PCA MLE',linestyle='--',linewidth=1.0)
    plt.axvline(z_s_mle,color='tab:green',label='Zhangjiashan:PCA MLE',linestyle='--',linewidth=0.5)

    ax3=plt.subplot(2,2,3)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Number of reduced predictors')
    plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("VMD-SVR",loc="left",)
    plt.text(14.38,0.988,"(c)",fontweight='normal',fontsize=7)
    plt.xticks(np.arange(0,16,1))
    labels=['Huaxian','Xianyang','Zhangjiashan']
    plt.plot(t,h_v_r2[1:],marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_v_r2[1:],marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_v_r2[1:],marker='v',label="Zhangjiashan",markerfacecolor='w')
    print("h_v_r2:{}".format(h_v_r2))
    print("x_v_r2:{}".format(x_v_r2))
    print("z_v_r2:{}".format(z_v_r2))
    plt.axhline(h_v_r2[0],color='tab:blue',label='Huaxian:without PCA',linestyle='-',linewidth=1.5)
    plt.axhline(x_v_r2[0],color='tab:orange',label='Xianyang:without PCA',linestyle='-',linewidth=1.0)
    plt.axhline(z_v_r2[0],color='tab:green',label='Zhangjiashan:without PCA',linestyle='-',linewidth=0.5)
    plt.axvline(h_v_mle,color='tab:blue',label='Huaxian:PCA MLE',linestyle='--',linewidth=1.5)
    plt.axvline(x_v_mle,color='tab:orange',label='Xianyang:PCA MLE',linestyle='--',linewidth=1.0)
    plt.axvline(z_v_mle,color='tab:green',label='Zhangjiashan:PCA MLE',linestyle='--',linewidth=0.5)
    
    plt.ylim([0.88,1])
    
 
    ax4=plt.subplot(2,2,4)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Number of reduced predictors')
    # plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("DWT-SVR",loc="left",)
    plt.text(14.38,0.996,"(d)",fontweight='normal',fontsize=7)
    plt.xticks(np.arange(0,16,1))
    plt.plot(t,h_w_r2[1:],marker='o',label="Huaxian:DWT-SVR",markerfacecolor='w')
    plt.plot(t,x_w_r2[1:],marker='s',label="Xianyang:DWT-SVR",markerfacecolor='w')
    plt.plot(t,z_w_r2[1:],marker='v',label="Zhangjiashan:DWT-SVR",markerfacecolor='w')
    plt.axhline(h_w_r2[0],color='tab:blue',label='Huaxian:without PCA',linestyle='-',linewidth=1.5)
    plt.axhline(x_w_r2[0],color='tab:orange',label='Xianyang:without PCA',linestyle='-',linewidth=1.0)
    plt.axhline(z_w_r2[0],color='tab:green',label='Zhangjiashan:without PCA',linestyle='-',linewidth=0.5)
    plt.axvline(h_w_mle,color='tab:blue',label='Huaxian:PCA MLE',linestyle='--',linewidth=1.5)
    plt.axvline(x_w_mle,color='tab:orange',label='Xianyang:PCA MLE',linestyle='--',linewidth=1.0)
    plt.axvline(z_w_mle,color='tab:green',label='Zhangjiashan:PCA MLE',linestyle='--',linewidth=0.5)
    print("h_w_r2:{}".format(h_w_r2))
    print("x_w_r2:{}".format(x_w_r2))
    print("z_w_r2:{}".format(z_w_r2))
    plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98,top=0.84, hspace=0.25, wspace=0.15)
    plt.savefig(graphs_path+"two_stage_pca_nse.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"two_stage_pca_nse.tif",format="TIFF",dpi=600)



    all_datas=[
        [
            [h_e_r2[1],x_e_r2[1],z_e_r2[1]],#reduced predictors 0
            [h_e_r2[2],x_e_r2[2],z_e_r2[2]],#reduced predictors 1
            [h_e_r2[3],x_e_r2[3],z_e_r2[3]],#reduced predictors 2
            [h_e_r2[4],x_e_r2[4],z_e_r2[4]],#reduced predictors 3
            [h_e_r2[5],x_e_r2[5],z_e_r2[5]],#reduced predictors 4
            [h_e_r2[6],x_e_r2[6],z_e_r2[6]],#reduced predictors 5
            [h_e_r2[7],x_e_r2[7],z_e_r2[7]],#reduced predictors 6
            [h_e_r2[8],x_e_r2[8],z_e_r2[8]],#reduced predictors 7
            [h_e_r2[9],x_e_r2[9],z_e_r2[9]],#reduced predictors 8
            [h_e_r2[10],x_e_r2[10],z_e_r2[10]],#reduced predictors 9
            [h_e_r2[11],x_e_r2[11],z_e_r2[11]],#reduced predictors 10
            [h_e_r2[12],x_e_r2[12],z_e_r2[12]],#reduced predictors 11
            [h_e_r2[13],x_e_r2[13],z_e_r2[13]],#reduced predictors 12
            [h_e_r2[14],x_e_r2[14],z_e_r2[14]],#reduced predictors 13
            [h_e_r2[15],x_e_r2[15],z_e_r2[15]],#reduced predictors 14
            [h_e_r2[16],x_e_r2[16],z_e_r2[16]],#reduced predictors 15
        ],

        [
            [h_s_r2[1],x_s_r2[1],z_s_r2[1]],#reduced predictors 0
            [h_s_r2[2],x_s_r2[2],z_s_r2[2]],#reduced predictors 1
            [h_s_r2[3],x_s_r2[3],z_s_r2[3]],#reduced predictors 2
            [h_s_r2[4],x_s_r2[4],z_s_r2[4]],#reduced predictors 3
            [h_s_r2[5],x_s_r2[5],z_s_r2[5]],#reduced predictors 4
            [h_s_r2[6],x_s_r2[6],z_s_r2[6]],#reduced predictors 5
            [h_s_r2[7],x_s_r2[7],z_s_r2[7]],#reduced predictors 6
            [h_s_r2[8],x_s_r2[8],z_s_r2[8]],#reduced predictors 7
            [h_s_r2[9],x_s_r2[9],z_s_r2[9]],#reduced predictors 8
            [h_s_r2[10],x_s_r2[10],z_s_r2[10]],#reduced predictors 9
            [h_s_r2[11],x_s_r2[11],z_s_r2[11]],#reduced predictors 10
            [h_s_r2[12],x_s_r2[12],z_s_r2[12]],#reduced predictors 11
            [h_s_r2[13],x_s_r2[13],z_s_r2[13]],#reduced predictors 12
            [h_s_r2[14],x_s_r2[14],z_s_r2[14]],#reduced predictors 13
            [h_s_r2[15],x_s_r2[15],z_s_r2[15]],#reduced predictors 14
            [h_s_r2[16],x_s_r2[16],z_s_r2[16]],#reduced predictors 15
        ],

        [
            [h_v_r2[1],x_v_r2[1],z_v_r2[1]],#reduced predictors 0
            [h_v_r2[2],x_v_r2[2],z_v_r2[2]],#reduced predictors 1
            [h_v_r2[3],x_v_r2[3],z_v_r2[3]],#reduced predictors 2
            [h_v_r2[4],x_v_r2[4],z_v_r2[4]],#reduced predictors 3
            [h_v_r2[5],x_v_r2[5],z_v_r2[5]],#reduced predictors 4
            [h_v_r2[6],x_v_r2[6],z_v_r2[6]],#reduced predictors 5
            [h_v_r2[7],x_v_r2[7],z_v_r2[7]],#reduced predictors 6
            [h_v_r2[8],x_v_r2[8],z_v_r2[8]],#reduced predictors 7
            [h_v_r2[9],x_v_r2[9],z_v_r2[9]],#reduced predictors 8
            [h_v_r2[10],x_v_r2[10],z_v_r2[10]],#reduced predictors 9
            [h_v_r2[11],x_v_r2[11],z_v_r2[11]],#reduced predictors 10
            [h_v_r2[12],x_v_r2[12],z_v_r2[12]],#reduced predictors 11
            [h_v_r2[13],x_v_r2[13],z_v_r2[13]],#reduced predictors 12
            [h_v_r2[14],x_v_r2[14],z_v_r2[14]],#reduced predictors 13
            [h_v_r2[15],x_v_r2[15],z_v_r2[15]],#reduced predictors 14
            [h_v_r2[16],x_v_r2[16],z_v_r2[16]],#reduced predictors 15
        ],

        [
            [h_w_r2[1],x_w_r2[1],z_w_r2[1]],#reduced predictors 0
            [h_w_r2[2],x_w_r2[2],z_w_r2[2]],#reduced predictors 1
            [h_w_r2[3],x_w_r2[3],z_w_r2[3]],#reduced predictors 2
            [h_w_r2[4],x_w_r2[4],z_w_r2[4]],#reduced predictors 3
            [h_w_r2[5],x_w_r2[5],z_w_r2[5]],#reduced predictors 4
            [h_w_r2[6],x_w_r2[6],z_w_r2[6]],#reduced predictors 5
            [h_w_r2[7],x_w_r2[7],z_w_r2[7]],#reduced predictors 6
            [h_w_r2[8],x_w_r2[8],z_w_r2[8]],#reduced predictors 7
            [h_w_r2[9],x_w_r2[9],z_w_r2[9]],#reduced predictors 8
            [h_w_r2[10],x_w_r2[10],z_w_r2[10]],#reduced predictors 9
            [h_w_r2[11],x_w_r2[11],z_w_r2[11]],#reduced predictors 10
            [h_w_r2[12],x_w_r2[12],z_w_r2[12]],#reduced predictors 11
            [h_w_r2[13],x_w_r2[13],z_w_r2[13]],#reduced predictors 12
            [h_w_r2[14],x_w_r2[14],z_w_r2[14]],#reduced predictors 13
            [h_w_r2[15],x_w_r2[15],z_w_r2[15]],#reduced predictors 14
            [h_w_r2[16],x_w_r2[16],z_w_r2[16]],#reduced predictors 15
        ],
    ]

    plt.figure(figsize=(5.51,4))
    x = list(range(16))
    x_s = [-0.38, 3.6, -0.38, 3.6]
    y_s = [0.9, 1.7, 3, 212]
    fig_ids = ['(a)', '(b)', '(c)', '(d)']
    # labels=['0']
    for i in range(len(all_datas)):
        ax1 = plt.subplot(2, 2, i+1)
        ax1.yaxis.grid(True)
        vplot1 = plt.violinplot(
            dataset=all_datas[i],
            positions=x,
            showmeans=True,
        )
        plt.ylabel(r'$NSE$')
        # plt.xticks(x, labels, rotation=45)
        plt.text(x_s[i], y_s[i], fig_ids[i], fontweight='normal', fontsize=7)
        for pc in vplot1['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96,top=0.98, hspace=0.6, wspace=0.45)
    plt.savefig(graphs_path+'/two_stage_pca_violin.eps',
                format='EPS', dpi=2000)
    plt.savefig(graphs_path+'/two_stage_pca_violin.tif',
                format='TIFF', dpi=1200)
    plt.show()



    

