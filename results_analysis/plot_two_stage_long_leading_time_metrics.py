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
from results_reader import read_long_leading_time
from fit_line import compute_linear_fit,compute_list_linear_fit


if __name__ == "__main__":
    # read metrics of vmd-svr models for all three stations
    h_v_records,h_v_predictions,h_v_r2,h_v_nrmse,h_v_mae,h_v_mape,h_v_ppts=read_long_leading_time(mode=None,station="Huaxian",decomposer="vmd")
    x_v_records,x_v_predictions,x_v_r2,x_v_nrmse,x_v_mae,x_v_mape,x_v_ppts=read_long_leading_time(mode=None,station="Xianyang",decomposer="vmd")
    z_v_records,z_v_predictions,z_v_r2,z_v_nrmse,z_v_mae,z_v_mape,z_v_ppts=read_long_leading_time(mode=None,station="Zhangjiashan",decomposer="vmd")

    # read metrics of eemd-svr models for all three stations
    h_e_records,h_e_predictions,h_e_r2,h_e_nrmse,h_e_mae,h_e_mape,h_e_ppts=read_long_leading_time(mode='0.3',station="Huaxian",decomposer="eemd")
    x_e_records,x_e_predictions,x_e_r2,x_e_nrmse,x_e_mae,x_e_mape,x_e_ppts=read_long_leading_time(mode='0.3',station="Xianyang",decomposer="eemd")
    z_e_records,z_e_predictions,z_e_r2,z_e_nrmse,z_e_mae,z_e_mape,z_e_ppts=read_long_leading_time(mode='0.3',station="Zhangjiashan",decomposer="eemd")

    # read metrics of ssa-svr models for all three stations
    h_s_records,h_s_predictions,h_s_r2,h_s_nrmse,h_s_mae,h_s_mape,h_s_ppts=read_long_leading_time(mode='0.3',station="Huaxian",decomposer="ssa")
    x_s_records,x_s_predictions,x_s_r2,x_s_nrmse,x_s_mae,x_s_mape,x_s_ppts=read_long_leading_time(mode='0.3',station="Xianyang",decomposer="ssa")
    z_s_records,z_s_predictions,z_s_r2,z_s_nrmse,z_s_mae,z_s_mape,z_s_ppts=read_long_leading_time(mode='0.3',station="Zhangjiashan",decomposer="ssa")

    # read metrics of dwt-svr models for all three stations
    h_w_records,h_w_predictions,h_w_r2,h_w_nrmse,h_w_mae,h_w_mape,h_w_ppts=read_long_leading_time(mode='0.3',station="Huaxian",decomposer="dwt")
    x_w_records,x_w_predictions,x_w_r2,x_w_nrmse,x_w_mae,x_w_mape,x_w_ppts=read_long_leading_time(mode='0.3',station="Xianyang",decomposer="dwt")
    z_w_records,z_w_predictions,z_w_r2,z_w_nrmse,z_w_mae,z_w_mape,z_w_ppts=read_long_leading_time(mode='0.3',station="Zhangjiashan",decomposer="dwt")

    
    
    t=[1,3,5,7,9]
    plt.figure(figsize=(7.48,5))
    ax1=plt.subplot(2,2,1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("VMD-SVR",loc="left",)
    plt.text(7.8,0.97,"VMD-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_v_r2,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_v_r2,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_v_r2,marker='v',label="Zhangjiashan",markerfacecolor='w')
    plt.legend(
        loc='upper left',
        # loc=0,
        # bbox_to_anchor=(0.08,1.01, 1,0.101),
        bbox_to_anchor=(0.6,1.18),
        # bbox_transform=plt.gcf().transFigure,
        ncol=3,
        shadow=False,
        frameon=True,
    )
    ax2=plt.subplot(2,2,2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("EEMD-SVR",loc="left",)
    plt.text(7.6,0.535,"EEMD-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_e_r2,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_e_r2,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_e_r2,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.legend()
    ax3=plt.subplot(2,2,3)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("SSA-SVR",loc="left",)
    plt.text(7.8,0.94,"SSA-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_s_r2,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_s_r2,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_s_r2,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.legend()
    ax4=plt.subplot(2,2,4)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("DWT-SVR",loc="left",)
    plt.text(7.8,0.95,"DWT-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_w_r2,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_w_r2,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_w_r2,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,top=0.94, hspace=0.3, wspace=0.25)
    plt.savefig(graphs_path+"two_stage_leading_time_nse_new.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"two_stage_leading_time_nse_new.tif",format="TIFF",dpi=600)
    # plt.show()

    t=[1,3,5,7,9]
    plt.figure(figsize=(7.48,5))
    ax1=plt.subplot(2,2,1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NRMSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("VMD-SVR",loc="left",)
    plt.text(7.8,0.97,"VMD-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_v_nrmse,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_v_nrmse,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_v_nrmse,marker='v',label="Zhangjiashan",markerfacecolor='w')
    plt.legend(
        loc='upper left',
        # loc=0,
        # bbox_to_anchor=(0.08,1.01, 1,0.101),
        bbox_to_anchor=(0.6,1.18),
        # bbox_transform=plt.gcf().transFigure,
        ncol=3,
        shadow=False,
        frameon=True,
    )
    ax2=plt.subplot(2,2,2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NRMSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("EEMD-SVR",loc="left",)
    plt.text(7.6,0.535,"EEMD-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_e_nrmse,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_e_nrmse,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_e_nrmse,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.legend()
    ax3=plt.subplot(2,2,3)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NRMSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("SSA-SVR",loc="left",)
    plt.text(7.8,0.94,"SSA-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_s_nrmse,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_s_nrmse,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_s_nrmse,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.legend()
    ax4=plt.subplot(2,2,4)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Leading time (month)')
    plt.ylabel(r'$NRMSE$')
    # plt.ylim(0.0,1.0)
    # plt.title("DWT-SVR",loc="left",)
    plt.text(7.8,0.95,"DWT-SVR")
    plt.xticks(np.arange(0,11,1))
    plt.plot(t,h_w_nrmse,marker='o',label="Huaxian",markerfacecolor='w')
    plt.plot(t,x_w_nrmse,marker='s',label="Xianyang",markerfacecolor='w')
    plt.plot(t,z_w_nrmse,marker='v',label="Zhangjiashan",markerfacecolor='w')
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98,top=0.94, hspace=0.3, wspace=0.25)
    plt.savefig(graphs_path+"two_stage_leading_time_nrmse_new.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"two_stage_leading_time_nrmse_new.tif",format="TIFF",dpi=600)

    def autolabels(rects,ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            height = round(height,2)
            ax.text(
                x=rect.get_x() + rect.get_width() / 2,
                y=height,
                s='{}'.format(height),
                fontsize=6,
                rotation=90,
                ha='center', va='bottom',
                        )

    

    values_list=[
        [h_v_r2,h_s_r2,h_e_r2,h_w_r2],
        [h_v_nrmse,h_s_nrmse,h_e_nrmse,h_w_nrmse],
        [h_v_ppts,h_s_ppts,h_e_ppts,h_w_ppts],
        [x_v_r2,x_s_r2,x_e_r2,x_w_r2],
        [x_v_nrmse,x_s_nrmse,x_e_nrmse,x_w_nrmse],
        [x_v_ppts,x_s_ppts,x_e_ppts,x_w_ppts],
        [z_v_r2,z_s_r2,z_e_r2,z_w_r2],
        [z_v_nrmse,z_s_nrmse,z_e_nrmse,z_w_nrmse],
        [z_v_ppts,z_s_ppts,z_e_ppts,z_w_ppts],
    ]

    pos=[1,3,5,7,9]
    labels=['1','3','5','7','9']
    width=0.45
    ylabels=[
        r"$NSE$",r"$NRMSE$",r"$PPTS(5)(\%)$",
        r"$NSE$",r"$NRMSE$",r"$PPTS(5)(\%)$",
        r"$NSE$",r"$NRMSE$",r"$PPTS(5)(\%)$",
    ]
    ylims=[
        [0,1.2],
        [0,1.0],
        [0,80],
        [0,1.2],
        [0,1.0],
        [0,70],
        [0,1.2],
        [0,1.5],
        [0,75],
    ]
    fig_id=[
        '(a1)','(a2)','(a3)',
        '(b1)','(b2)','(b3)',
        '(c1)','(c2)','(c3)',
    ]
    fig = plt.figure(figsize=(7.48,5))
    for i in range(len(values_list)):
        ax = fig.add_subplot(3,3,i+1)
        action=[-1,0,1,2]
        models=['VMD-SVR','SSA-SVR','EEMD-SVR','DWT-SVR',]
        for j in range(len(values_list[i])):
            bars = ax.bar([p+action[j]*width for p in pos],values_list[i][j],width,alpha=0.75,label=models[j])
            autolabels(bars,ax)
        ax.set_ylim(ylims[i])
        ax.set_xlabel("Leading time(month)\n"+fig_id[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xticks([p+width/2 for p in pos])
        ax.set_xticklabels(labels)
        if i==1:
            plt.legend(
                loc='upper center',
                # loc=0,
                # bbox_to_anchor=(0.08,1.01, 1,0.101),
                bbox_to_anchor=(0.5,1.27),
                # bbox_transform=plt.gcf().transFigure,
                ncol=4,
                shadow=False,
                frameon=True,
            )
    plt.subplots_adjust(left=0.09, bottom=0.1, right=0.98,top=0.94, hspace=0.55, wspace=0.3)
    plt.savefig(graphs_path+"two_stage_leading_time_metrics_new.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"two_stage_leading_time_metrics_new.tif",format="TIFF",dpi=600)


###########################################################################################################
    records_list=[
        h_e_records,h_s_records,h_v_records,h_w_records,
        x_e_records,x_s_records,x_v_records,x_w_records,
        z_e_records,z_s_records,z_v_records,z_w_records,
    ]
    predictions_list=[
        h_e_predictions,h_s_predictions,h_v_predictions,h_w_predictions,
        x_e_predictions,x_s_predictions,x_v_predictions,x_w_predictions,
        z_e_predictions,z_s_predictions,z_v_predictions,z_w_predictions,
    ]
    x=[17,19,18,18,9.8,11.1,10.8,11.1,2.6,2.9,2.8,2.75,]
    y=[1.8,2,2,1.8,1.0,1.1,1.1,1.1,.2,.2,.2,.2]
    text=[
        'EEMD-SVR','SSA-SVR','VMD-SVR','DWT-SVR',
        'EEMD-SVR','SSA-SVR','VMD-SVR','DWT-SVR',
        'EEMD-SVR','SSA-SVR','VMD-SVR','DWT-SVR',
    ]
    fig_id=[
        '(a1)','(a2)','(a3)','(a4)',
        '(b1)','(b2)','(b3)','(b4)',
        '(c1)','(c2)','(c3)','(c4)',
    ]
    plt.figure(figsize=(7.48,5.8))
    for j in range(len(records_list)):
        ax=plt.subplot(3,4,j+1, aspect='equal')
        plt.text(x[j],y[j],fig_id[j]+text[j],fontsize=7)
        xx,linear_list,xymin,xymax=compute_list_linear_fit(
            records_list=records_list[j],
            predictions_list=predictions_list[j],
        )
        if j in [4,5,6,7]:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if j in range(8,12):
            plt.xlabel('Predictions(' + r'$10^8m^3$' +')', )
        if j in [0,4,8]:
            plt.ylabel('Records(' + r'$10^8m^3$' + ')', )
        models=['1-month','3-month','5-month','7-month','9-month']
        markers=['o','v','*','s','+',]
        zorders=[4,3,2,1,0]
        for i in range(len(predictions_list[j])):
            print("length of predictions list:{}".format(len(predictions_list[j])))
            print("length of records list:{}".format(len(records_list[j])))
            # plt.plot(predictions_list[i], records_list[i],marker=markers[i], markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
            plt.scatter(predictions_list[j][i], records_list[j][i],marker=markers[i],zorder=zorders[i])
            plt.plot(xx, linear_list[i], '--', label=models[i],linewidth=1.0,zorder=zorders[i])
        plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
        plt.xlim([xymin,xymax])
        plt.ylim([xymin,xymax])
        if j==0:
            plt.legend(
                        loc='upper left',
                        # bbox_to_anchor=(0.08,1.01, 1,0.101),
                        bbox_to_anchor=(0.8,1.15),
                        ncol=6,
                        shadow=False,
                        frameon=True,
                        # fontsize=6,
                        )
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.99,top=0.96, hspace=0.15, wspace=0.05)
    plt.savefig(graphs_path+"two_stage_leading_time_scatters.eps",format="EPS",dpi=2000)
    plt.savefig(graphs_path+"two_stage_leading_time_scatters.tif",format="TIFF",dpi=600)


    nse_eemd_data1=[h_e_r2[0],x_e_r2[0],z_e_r2[0]]
    nse_eemd_data3=[h_e_r2[1],x_e_r2[1],z_e_r2[1]]
    nse_eemd_data5=[h_e_r2[2],x_e_r2[2],z_e_r2[2]]
    nse_eemd_data7=[h_e_r2[3],x_e_r2[3],z_e_r2[3]]
    nse_eemd_data9=[h_e_r2[4],x_e_r2[4],z_e_r2[4]]
    nse_ssa_data1=[h_s_r2[0],x_s_r2[0],z_s_r2[0]]
    nse_ssa_data3=[h_s_r2[1],x_s_r2[1],z_s_r2[1]]
    nse_ssa_data5=[h_s_r2[2],x_s_r2[2],z_s_r2[2]]
    nse_ssa_data7=[h_s_r2[3],x_s_r2[3],z_s_r2[3]]
    nse_ssa_data9=[h_s_r2[4],x_s_r2[4],z_s_r2[4]]
    nse_vmd_data1=[h_v_r2[0],x_v_r2[0],z_v_r2[0]]
    nse_vmd_data3=[h_v_r2[1],x_v_r2[1],z_v_r2[1]]
    nse_vmd_data5=[h_v_r2[2],x_v_r2[2],z_v_r2[2]]
    nse_vmd_data7=[h_v_r2[3],x_v_r2[3],z_v_r2[3]]
    nse_vmd_data9=[h_v_r2[4],x_v_r2[4],z_v_r2[4]]
    nse_dwt_data1=[h_w_r2[0],x_w_r2[0],z_w_r2[0]]
    nse_dwt_data3=[h_w_r2[1],x_w_r2[1],z_w_r2[1]]
    nse_dwt_data5=[h_w_r2[2],x_w_r2[2],z_w_r2[2]]
    nse_dwt_data7=[h_w_r2[3],x_w_r2[3],z_w_r2[3]]
    nse_dwt_data9=[h_w_r2[4],x_w_r2[4],z_w_r2[4]]
    nse_data=[
        nse_eemd_data1,
        nse_eemd_data3,
        nse_eemd_data5,
        nse_eemd_data7,
        nse_eemd_data9,
        nse_ssa_data1,
        nse_ssa_data3,
        nse_ssa_data5,
        nse_ssa_data7,
        nse_ssa_data9,
        nse_vmd_data1,
        nse_vmd_data3,
        nse_vmd_data5,
        nse_vmd_data7,
        nse_vmd_data9,
        nse_dwt_data1,
        nse_dwt_data3,
        nse_dwt_data5,
        nse_dwt_data7,
        nse_dwt_data9,
    ]

    eemd_mean_nse=[
        sum(nse_eemd_data1)/len(nse_eemd_data1),
        sum(nse_eemd_data3)/len(nse_eemd_data3),
        sum(nse_eemd_data5)/len(nse_eemd_data5),
        sum(nse_eemd_data7)/len(nse_eemd_data7),
        sum(nse_eemd_data9)/len(nse_eemd_data9),
    ]

    ssa_mean_nse=[
        sum(nse_ssa_data1)/len(nse_ssa_data1),
        sum(nse_ssa_data3)/len(nse_ssa_data3),
        sum(nse_ssa_data5)/len(nse_ssa_data5),
        sum(nse_ssa_data7)/len(nse_ssa_data7),
        sum(nse_ssa_data9)/len(nse_ssa_data9),
    ]

    vmd_mean_nse=[
        sum(nse_vmd_data1)/len(nse_vmd_data1),
        sum(nse_vmd_data3)/len(nse_vmd_data3),
        sum(nse_vmd_data5)/len(nse_vmd_data5),
        sum(nse_vmd_data7)/len(nse_vmd_data7),
        sum(nse_vmd_data9)/len(nse_vmd_data9),
    ]

    dwt_mean_nse=[
        sum(nse_dwt_data1)/len(nse_dwt_data1),
        sum(nse_dwt_data3)/len(nse_dwt_data3),
        sum(nse_dwt_data5)/len(nse_dwt_data5),
        sum(nse_dwt_data7)/len(nse_dwt_data7),
        sum(nse_dwt_data9)/len(nse_dwt_data9),
    ]
    

    nrmse_eemd_data1=[h_e_nrmse[0],x_e_nrmse[0],z_e_nrmse[0]]
    nrmse_eemd_data3=[h_e_nrmse[1],x_e_nrmse[1],z_e_nrmse[1]]
    nrmse_eemd_data5=[h_e_nrmse[2],x_e_nrmse[2],z_e_nrmse[2]]
    nrmse_eemd_data7=[h_e_nrmse[3],x_e_nrmse[3],z_e_nrmse[3]]
    nrmse_eemd_data9=[h_e_nrmse[4],x_e_nrmse[4],z_e_nrmse[4]]
    nrmse_ssa_data1=[h_s_nrmse[0],x_s_nrmse[0],z_s_nrmse[0]]
    nrmse_ssa_data3=[h_s_nrmse[1],x_s_nrmse[1],z_s_nrmse[1]]
    nrmse_ssa_data5=[h_s_nrmse[2],x_s_nrmse[2],z_s_nrmse[2]]
    nrmse_ssa_data7=[h_s_nrmse[3],x_s_nrmse[3],z_s_nrmse[3]]
    nrmse_ssa_data9=[h_s_nrmse[4],x_s_nrmse[4],z_s_nrmse[4]]
    nrmse_vmd_data1=[h_v_nrmse[0],x_v_nrmse[0],z_v_nrmse[0]]
    nrmse_vmd_data3=[h_v_nrmse[1],x_v_nrmse[1],z_v_nrmse[1]]
    nrmse_vmd_data5=[h_v_nrmse[2],x_v_nrmse[2],z_v_nrmse[2]]
    nrmse_vmd_data7=[h_v_nrmse[3],x_v_nrmse[3],z_v_nrmse[3]]
    nrmse_vmd_data9=[h_v_nrmse[4],x_v_nrmse[4],z_v_nrmse[4]]
    nrmse_dwt_data1=[h_w_nrmse[0],x_w_nrmse[0],z_w_nrmse[0]]
    nrmse_dwt_data3=[h_w_nrmse[1],x_w_nrmse[1],z_w_nrmse[1]]
    nrmse_dwt_data5=[h_w_nrmse[2],x_w_nrmse[2],z_w_nrmse[2]]
    nrmse_dwt_data7=[h_w_nrmse[3],x_w_nrmse[3],z_w_nrmse[3]]
    nrmse_dwt_data9=[h_w_nrmse[4],x_w_nrmse[4],z_w_nrmse[4]]
    nrmse_data=[
        nrmse_eemd_data1,
        nrmse_eemd_data3,
        nrmse_eemd_data5,
        nrmse_eemd_data7,
        nrmse_eemd_data9,
        nrmse_ssa_data1,
        nrmse_ssa_data3,
        nrmse_ssa_data5,
        nrmse_ssa_data7,
        nrmse_ssa_data9,
        nrmse_vmd_data1,
        nrmse_vmd_data3,
        nrmse_vmd_data5,
        nrmse_vmd_data7,
        nrmse_vmd_data9,
        nrmse_dwt_data1,
        nrmse_dwt_data3,
        nrmse_dwt_data5,
        nrmse_dwt_data7,
        nrmse_dwt_data9,
    ]

    eemd_mean_nrmse=[
        sum(nrmse_eemd_data1)/len(nrmse_eemd_data1),
        sum(nrmse_eemd_data3)/len(nrmse_eemd_data3),
        sum(nrmse_eemd_data5)/len(nrmse_eemd_data5),
        sum(nrmse_eemd_data7)/len(nrmse_eemd_data7),
        sum(nrmse_eemd_data9)/len(nrmse_eemd_data9),
    ]

    ssa_mean_nrmse=[
        sum(nrmse_ssa_data1)/len(nrmse_ssa_data1),
        sum(nrmse_ssa_data3)/len(nrmse_ssa_data3),
        sum(nrmse_ssa_data5)/len(nrmse_ssa_data5),
        sum(nrmse_ssa_data7)/len(nrmse_ssa_data7),
        sum(nrmse_ssa_data9)/len(nrmse_ssa_data9),
    ]

    vmd_mean_nrmse=[
        sum(nrmse_vmd_data1)/len(nrmse_vmd_data1),
        sum(nrmse_vmd_data3)/len(nrmse_vmd_data3),
        sum(nrmse_vmd_data5)/len(nrmse_vmd_data5),
        sum(nrmse_vmd_data7)/len(nrmse_vmd_data7),
        sum(nrmse_vmd_data9)/len(nrmse_vmd_data9),
    ]

    dwt_mean_nrmse=[
        sum(nrmse_dwt_data1)/len(nrmse_dwt_data1),
        sum(nrmse_dwt_data3)/len(nrmse_dwt_data3),
        sum(nrmse_dwt_data5)/len(nrmse_dwt_data5),
        sum(nrmse_dwt_data7)/len(nrmse_dwt_data7),
        sum(nrmse_dwt_data9)/len(nrmse_dwt_data9),
    ]

    ppts_eemd_data1=[h_e_ppts[0],x_e_ppts[0],z_e_ppts[0]]
    ppts_eemd_data3=[h_e_ppts[1],x_e_ppts[1],z_e_ppts[1]]
    ppts_eemd_data5=[h_e_ppts[2],x_e_ppts[2],z_e_ppts[2]]
    ppts_eemd_data7=[h_e_ppts[3],x_e_ppts[3],z_e_ppts[3]]
    ppts_eemd_data9=[h_e_ppts[4],x_e_ppts[4],z_e_ppts[4]]
    ppts_ssa_data1=[h_s_ppts[0],x_s_ppts[0],z_s_ppts[0]]
    ppts_ssa_data3=[h_s_ppts[1],x_s_ppts[1],z_s_ppts[1]]
    ppts_ssa_data5=[h_s_ppts[2],x_s_ppts[2],z_s_ppts[2]]
    ppts_ssa_data7=[h_s_ppts[3],x_s_ppts[3],z_s_ppts[3]]
    ppts_ssa_data9=[h_s_ppts[4],x_s_ppts[4],z_s_ppts[4]]
    ppts_vmd_data1=[h_v_ppts[0],x_v_ppts[0],z_v_ppts[0]]
    ppts_vmd_data3=[h_v_ppts[1],x_v_ppts[1],z_v_ppts[1]]
    ppts_vmd_data5=[h_v_ppts[2],x_v_ppts[2],z_v_ppts[2]]
    ppts_vmd_data7=[h_v_ppts[3],x_v_ppts[3],z_v_ppts[3]]
    ppts_vmd_data9=[h_v_ppts[4],x_v_ppts[4],z_v_ppts[4]]
    ppts_dwt_data1=[h_w_ppts[0],x_w_ppts[0],z_w_ppts[0]]
    ppts_dwt_data3=[h_w_ppts[1],x_w_ppts[1],z_w_ppts[1]]
    ppts_dwt_data5=[h_w_ppts[2],x_w_ppts[2],z_w_ppts[2]]
    ppts_dwt_data7=[h_w_ppts[3],x_w_ppts[3],z_w_ppts[3]]
    ppts_dwt_data9=[h_w_ppts[4],x_w_ppts[4],z_w_ppts[4]]
    ppts_data=[
        ppts_eemd_data1,
        ppts_eemd_data3,
        ppts_eemd_data5,
        ppts_eemd_data7,
        ppts_eemd_data9,
        ppts_ssa_data1,
        ppts_ssa_data3,
        ppts_ssa_data5,
        ppts_ssa_data7,
        ppts_ssa_data9,
        ppts_vmd_data1,
        ppts_vmd_data3,
        ppts_vmd_data5,
        ppts_vmd_data7,
        ppts_vmd_data9,
        ppts_dwt_data1,
        ppts_dwt_data3,
        ppts_dwt_data5,
        ppts_dwt_data7,
        ppts_dwt_data9,
    ]

    eemd_mean_ppts=[
        sum(ppts_eemd_data1)/len(ppts_eemd_data1),
        sum(ppts_eemd_data3)/len(ppts_eemd_data3),
        sum(ppts_eemd_data5)/len(ppts_eemd_data5),
        sum(ppts_eemd_data7)/len(ppts_eemd_data7),
        sum(ppts_eemd_data9)/len(ppts_eemd_data9),
    ]

    ssa_mean_ppts=[
        sum(ppts_ssa_data1)/len(ppts_ssa_data1),
        sum(ppts_ssa_data3)/len(ppts_ssa_data3),
        sum(ppts_ssa_data5)/len(ppts_ssa_data5),
        sum(ppts_ssa_data7)/len(ppts_ssa_data7),
        sum(ppts_ssa_data9)/len(ppts_ssa_data9),
    ]

    vmd_mean_ppts=[
        sum(ppts_vmd_data1)/len(ppts_vmd_data1),
        sum(ppts_vmd_data3)/len(ppts_vmd_data3),
        sum(ppts_vmd_data5)/len(ppts_vmd_data5),
        sum(ppts_vmd_data7)/len(ppts_vmd_data7),
        sum(ppts_vmd_data9)/len(ppts_vmd_data9),
    ]

    dwt_mean_ppts=[
        sum(ppts_dwt_data1)/len(ppts_dwt_data1),
        sum(ppts_dwt_data3)/len(ppts_dwt_data3),
        sum(ppts_dwt_data5)/len(ppts_dwt_data5),
        sum(ppts_dwt_data7)/len(ppts_dwt_data7),
        sum(ppts_dwt_data9)/len(ppts_dwt_data9),
    ]


    nse_lines=[
        eemd_mean_nse,
        ssa_mean_nse,
        vmd_mean_nse,
        dwt_mean_nse,
    ]

    nrmse_lines=[
        eemd_mean_nrmse,
        ssa_mean_nrmse,
        vmd_mean_nrmse,
        dwt_mean_nrmse,
    ]

    ppts_lines=[
        eemd_mean_ppts,
        ssa_mean_ppts,
        vmd_mean_ppts,
        dwt_mean_ppts,
    ]

    lines=[
        nse_lines,
        nrmse_lines,
        ppts_lines,
    ]
    
    

    all_datas = [nse_data,nrmse_data,ppts_data]
    fig_index=["(a)","(b)","(c)"]
    labels=[
        "EEMD-SVR\n(1-month ahead)",
        "EEMD-SVR\n(3-month ahead)",
        "EEMD-SVR\n(5-month ahead)",
        "EEMD-SVR\n(7-month ahead)",
        "EEMD-SVR\n(9-month ahead)",
        "SSA-SVR\n(1-month ahead)",
        "SSA-SVR\n(3-month ahead)",
        "SSA-SVR\n(5-month ahead)",
        "SSA-SVR\n(7-month ahead)",
        "SSA-SVR\n(9-month ahead)",
        "VMD-SVR\n(1-month ahead)",
        "VMD-SVR\n(3-month ahead)",
        "VMD-SVR\n(5-month ahead)",
        "VMD-SVR\n(7-month ahead)",
        "VMD-SVR\n(9-month ahead)",
        "DWT-SVR\n(1-month ahead)",
        "DWT-SVR\n(3-month ahead)",
        "DWT-SVR\n(5-month ahead)",
        "DWT-SVR\n(7-month ahead)",
        "DWT-SVR\n(9-month ahead)",
        ]
    x = list(range(20))
    ylabels=[
        r"$NSE$",r"$NRMSE$",r"$PPTS(5)(\%)$",
    ]
    x_s=[-1.1,-1.1,-1.1]
    y_s=[0.93,1.36,70]
    plt.figure(figsize=(7.48, 5.54))
    for i in range(len(all_datas)):
        ax1 = plt.subplot(3, 1, i+1)
        ax1.yaxis.grid(True)
        ax1.text(x_s[i],y_s[i],fig_index[i],fontsize=7)
        vplot1 = plt.violinplot(
            dataset=all_datas[i],
            positions=x,
            showmeans=True,
        )
        ax1.plot(list(range(0,5)),lines[i][0],'--',lw=0.5,color='blue')
        ax1.plot(list(range(5,10)),lines[i][1],'--',lw=0.5,color='blue')
        ax1.plot(list(range(10,15)),lines[i][2],'--',lw=0.5,color='blue')
        ax1.plot(list(range(15,20)),lines[i][3],'--',lw=0.5,color='blue')
        print(type(vplot1["cmeans"]))
        plt.ylabel(ylabels[i])
        if i==len(all_datas)-1:
            plt.xticks(x, labels, rotation=45)
        else:
            plt.xticks([])
        for pc in vplot1['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
   
    plt.tight_layout()
    plt.savefig(graphs_path+'/long_leading_time_metrics_violin.eps',
                format='EPS', dpi=2000)
    plt.savefig(graphs_path+'/long_leading_time_metrics_violin.tif',
                format='TIFF', dpi=1200)
    plt.show()

    print("NSE"+"-"*100)
    base_nse = vmd_mean_nse[0]
    for i in range(1,len(eemd_mean_nse)):
        ratio = (vmd_mean_nse[i]-base_nse)/base_nse*100
        print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (ssa_mean_nse[i]-base_nse)/base_nse*100
        print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (eemd_mean_nse[i]-base_nse)/base_nse*100
        print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (dwt_mean_nse[i]-base_nse)/base_nse*100
        print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    print("NRMSE"+"-"*100)
    base_nrmse = vmd_mean_nrmse[0]
    for i in range(1,len(eemd_mean_nrmse)):
        ratio = (vmd_mean_nrmse[i]-base_nrmse)/base_nrmse*100
        print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (ssa_mean_nrmse[i]-base_nrmse)/base_nrmse*100
        print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (eemd_mean_nrmse[i]-base_nrmse)/base_nrmse*100
        print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (dwt_mean_nrmse[i]-base_nrmse)/base_nrmse*100
        print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
    print("PPTS"+"-"*100)
    base_ppts = vmd_mean_ppts[0]
    for i in range(1,len(eemd_mean_ppts)):
        ratio = (vmd_mean_ppts[i]-base_ppts)/base_ppts*100
        print("VMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (ssa_mean_ppts[i]-base_ppts)/base_ppts*100
        print("SSA-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (eemd_mean_ppts[i]-base_ppts)/base_ppts*100
        print("EEMMD-SVR for {}-month reduced:{}%".format(2*i+1,ratio))
        ratio = (dwt_mean_ppts[i]-base_ppts)/base_ppts*100
        print("DWT-SVR for {}-month reduced:{}%".format(2*i+1,ratio))

