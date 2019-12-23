import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence,plot_objective,plot_evaluations
import deprecated

# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=7
plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=2

def plot_relation(records, predictions, fig_savepath, figsize=(12,4),fontsize=10,log_reconvert=True, format='PNG',dpi=300):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1). 
    """
    if log_reconvert:
        records = np.power(10, records / 2.3) - 1
        predictions = np.array(list(predictions))
        predictions = np.power(10, predictions / 2.3) - 1
    else:
        predictions = np.array(list(predictions))

    pred_min =predictions.min()
    pred_max = predictions.max()
    record_min = records.min()
    record_max = records.max()
    if pred_min<record_min:
        xymin = pred_min
    else:
        xymin = record_min
    if pred_max>record_max:
        xymax = pred_max
    else:
        xymax=record_max
    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    # print('a:{}'.format(coeff[0]))
    # print('b:{}'.format(coeff[1]))
    plt.figure(num=1, figsize=figsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=fontsize)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=fontsize)
    # plt.plot(predictions, records, 'o', color='blue', label='',markersize=6.5)
    plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    # plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    # plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    plt.legend(loc=0, shadow=True, fontsize=fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_pred(records, predictions, fig_savepath,xlabel='Time(month)',figsize=(12, 4),fontsize=10,log_reconvert=True,format='PNG',dpi=300):
    """
    Plot lines of records and predictions.
    Args:
        records: record data set.
        predictions: prediction data set.
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1).
    """

    length = records.size
    t = np.linspace(start=1, stop=length, num=length)

    if log_reconvert:
        records = np.power(10, records / 2.3) - 1
        predictions = np.array(list(predictions))
        predictions = np.power(10, predictions / 2.3) - 1
    else:
        predictions = np.array(list(predictions))

    plt.figure(figsize=figsize)
    # plt.title('flow prediction based on DNN')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=fontsize)
    plt.plot(t, records, '-', color='blue', label='records')
    plt.plot(t, predictions, '--', color='red', label='predictions')
    plt.legend(loc='upper left', shadow=True, fontsize=fontsize)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()



def plot_normreconvert_relation(records, predictions,series_max,series_min,fig_savepath,figsize=(12, 4),fontsize=10,format='PNG',dpi=300):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model.
        series_mean: Datafram contains mean value of features and labels.
        series_max: Dataframe contains max value of features and labels.
        series_min: Datafram coontains min value of features and labels.
        fig_savepath: the path where the plot figure will be saved.
    """

    # records = np.multiply(records,series_max["Y"]-series_min["Y"])+series_mean["Y"]
    # records = np.multiply(records + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    # predictions = np.array(list(predictions))
    # predictions = np.multiply(predictions, series_max["Y"] - series_min["Y"]) + series_mean["Y"]
    # predictions = np.multiply(predictions + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]


    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * predictions + coeff[1]
    ideal_fit = 1 * predictions

    # compare the records and predictions
    plt.figure(num=1, figsize=figsize)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    # plt.title('The relationship between records and predictions').
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=fontsize)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=fontsize)
    # plt.plot(predictions, records, 'o', color='blue', label='', linewidth=1.0)
    plt.plot(predictions, records, 'o', color='',edgecolor='blue', label='', linewidth=1.0)
    plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit')
    plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit')
    # plt.text(24,28,'y={:.2f}'.format(coeff[0])+'*x{:.2f}'.format(coeff[1]),fontsize=fontsize)
    # plt.text(26,24,'y=1.0*x',fontsize=fontsize)
    plt.legend(loc='upper left', shadow=True, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()


def plot_normreconvert_pred(records, predictions,series_max,series_min, fig_savepath,xlabel='Time(month)',figsize=(12, 4),fontsize=10,format='PNG',dpi=300):
    """
    Plot lines of records and predictions.
    Args:
        records: record data set.
        predictions: prediction data set.
        fig_savepath: the path where the plot figure will be saved.
        log_reconvert: If ture, reconvert the records and prediction to orignial data by G=10^(z/2.3)-1.
        where 'G' is the original dataset and 'z' is the transformed data by z = 2.3*log_10(G+1).
    """

    length = records.size
    t = np.linspace(start=1, stop=length, num=length)

    # records = np.multiply(records,series_max["Y"]-series_min["Y"])+series_mean["Y"]
    # records = np.multiply(records + 1, series_max["Y"] - series_min["Y"]) / 2 + series_min["Y"]
    # predictions = np.array(list(predictions))
    # predictions = np.multiply(predictions + 1, series_max["Y"] -eries_min["Y"]) / 2 + series_min["Y"]

    plt.figure(figsize=figsize)
    # plt.title('flow prediction based on DNN')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time(month)', fontsize=fontsize)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=fontsize)
    plt.plot(t, records, '-', color='blue', label='records')
    plt.plot(t, predictions, '--', color='red', label='predictions')
    plt.legend(loc='upper left', shadow=True, fontsize=fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.3)
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()

def plot_rela_pred(records, predictions, fig_savepath,xlabel='Time(month)',figsize=(12, 4),fontsize=10,format='PNG',dpi=300):
    """ 
    Plot the relations between the records and predictions.
    Args:
        records: the actual measured records.
        predictions: the predictions obtained by model
        fig_savepath: the path where the plot figure will be saved.
    """
    length = records.size
    t = np.linspace(start=1, stop=length, num=length)
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=fontsize)
    plt.plot(t, records, '-', color='blue', label='records',linewidth=1.0)
    plt.plot(t, predictions, '--', color='red', label='predictions',linewidth=1.0)
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.005,1.2),
        shadow=False,
        frameon=False,
        fontsize=fontsize)
    plt.subplot(1, 2, 2)
    pred_min =predictions.min()
    pred_max = predictions.max()
    record_min = records.min()
    record_max = records.max()
    if pred_min<record_min:
        xymin = pred_min
    else:
        xymin = record_min
    if pred_max>record_max:
        xymax = pred_max
    else:
        xymax=record_max
    xx = np.arange(start=xymin,stop=xymax+1,step=1.0) 
    coeff = np.polyfit(predictions, records, 1)
    linear_fit = coeff[0] * xx + coeff[1]
    # print('a:{}'.format(coeff[0]))
    # print('b:{}'.format(coeff[1]))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('predictions(' + r'$m^3$' + '/s)', fontsize=fontsize)
    plt.ylabel('records(' + r'$m^3$' + '/s)', fontsize=fontsize)
    # plt.plot(predictions, records, 'o', color='blue', label='',markersize=6.5)
    plt.plot(predictions, records,'o', markerfacecolor='w',markeredgecolor='blue',markersize=6.5)
    # plt.plot(predictions, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    # plt.plot(predictions, ideal_fit, '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.plot(xx, linear_fit, '--', color='red', label='Linear fit',linewidth=1.0)
    plt.plot([xymin,xymax], [xymin,xymax], '-', color='black', label='Ideal fit',linewidth=1.0)
    plt.xlim([xymin,xymax])
    plt.ylim([xymin,xymax])
    plt.legend(
        # loc='upper left',
        loc=0,
        # bbox_to_anchor=(0.05,1),
        shadow=False,
        frameon=False,
        fontsize=fontsize)
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()

def plot_history(history,path1,path2,format='PNG',dpi=300):
    hist = pd.DataFrame(history.history)
    hist['epoch']=history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean abs error')
    plt.plot(hist['epoch'],hist['mean_absolute_error'],label='Train error')
    plt.plot(hist['epoch'],hist['val_mean_absolute_error'],label='Val error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.savefig(path1, format=format, dpi=dpi)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    # plt.ylim([0,0.04])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path2, format=format, dpi=dpi)
    # plt.show()

def plot_error_distribution(records,predictions,fig_savepath,format='PNG',dpi=300):
    """
    Plot error distribution from the predictions and labels.
    Args:
        predictions: prdictions obtained by ml 
        records: real records
        fig_savepath: path to save a error distribution figure

    Return
        A figure of error distribution
    """
    plt.figure()
    error = predictions - records
    # plt.hist(error,bins=25)
    plt.hist(error, 50, density=True,log=True, facecolor='g', alpha=0.75)
    plt.xlabel('Prediction Error')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(fig_savepath, format=format, dpi=dpi)
    # plt.show()

def plot_convergence_(OptimizeResult,fig_savepath,format='PNG',dpi=300):
    """
    Plot one or several convergence traces.
    Parameters
    args[i] [OptimizeResult, list of OptimizeResult, or tuple]: The result(s) for which to plot the convergence trace.
    
    if OptimizeResult, then draw the corresponding single trace;
    if list of OptimizeResult, then draw the corresponding convergence traces in transparency, along with the average convergence trace;
    if tuple, then args[i][0] should be a string label and args[i][1] an OptimizeResult or a list of OptimizeResult.
    ax [Axes, optional]: The matplotlib axes on which to draw the plot, or None to create a new one.
    
    true_minimum [float, optional]: The true minimum value of the function, if known.
    
    yscale [None or string, optional]: The scale for the y-axis.

    fig_savepath [string]: The path to save a convergence figure
    
    Returns
    ax: [Axes]: The matplotlib axes.
    """
    fig = plt.figure(num=1,figsize=(12,12))
    ax0 = plt.gca()
    plot_convergence(OptimizeResult,ax=ax0, true_minimum=0.0,)
    # ax0.set_ylabel(r"$\min MAE$ after $n$ calls")
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath,format=format,dpi=dpi)
    # plt.show()

def plot_evaluations_(OptimizeResult,dimensions,fig_savepath,format='PNG',dpi=300):
    """
    Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search space and in which order samples were evaluated. Pairwise scatter plots are shown on the off-diagonal for each dimension of the search space. The order in which samples were evaluated is encoded in each point's color. The diagonal shows a histogram of sampled values for each dimension. A red point indicates the found minimum.

    Note: search spaces that contain Categorical dimensions are currently not supported by this function.

    Parameters
    result [OptimizeResult] The result for which to create the scatter plot matrix.

    bins [int, bins=20]: Number of bins to use for histograms on the diagonal.

    dimensions [list of str, default=None] Labels of the dimension variables. None defaults to space.dimensions[i].name, or if also None to ['X_0', 'X_1', ..].

    fig_savepath [string]: The path to save a convergence figure

    Returns
    ax: [Axes]: The matplotlib axes.
    """
    fig = plt.figure(num=1,figsize=(12,12))
    ax0 = plt.gca()
    plot_evaluations(OptimizeResult,dimensions=dimensions)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath,format=format,dpi=dpi)
    # plt.show()

def plot_objective_(OptimizeResult,dimensions,fig_savepath,figsize=(12,12),format='PNG',dpi=300):
    """
    Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension i with respect to the objective function. The off-diagonal shows the partial dependence for dimensions i and j with respect to the objective function. The objective function is approximated by result.model.
    
    Pairwise scatter plots of the points at which the objective function was directly evaluated are shown on the off-diagonal. A red point indicates the found minimum.
    
    Note: search spaces that contain Categorical dimensions are currently not supported by this function.
    
    Parameters
    result [OptimizeResult] The result for which to create the scatter plot matrix.
    
    levels [int, default=10] Number of levels to draw on the contour plot, passed directly to plt.contour().
    
    n_points [int, default=40] Number of points at which to evaluate the partial dependence along each dimension.
    
    n_samples [int, default=250] Number of random samples to use for averaging the model function at each of the n_points.
    
    size [float, default=2] Height (in inches) of each facet.
    
    zscale [str, default='linear'] Scale to use for the z axis of the contour plots. Either 'linear' or 'log'.
    
    dimensions [list of str, default=None] Labels of the dimension variables. None defaults to space.dimensions[i].name, or if also None to ['X_0', 'X_1', ..].

    fig_savepath [string]: The path to save a convergence figure
    
    Returns
    ax: [Axes]: The matplotlib axes.
    """
    fig = plt.figure(num=1,figsize=figsize)
    ax0 = plt.gca()
    _ = plot_objective(OptimizeResult,dimensions=dimensions)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.98, hspace=0.1, wspace=0.2)
    plt.savefig(fig_savepath,format=format,dpi=dpi)
    # plt.show()

def plot_subsignals_pred(predictions,records,test_len,full_len,fig_savepath,subsignals_name=None,figsize=(7.48,7.48),fontsize=10,format='PNG',dpi=300,xlabel='Time(month)'):
    assert predictions.shape[1]==records.shape[1]

    if subsignals_name==None:
        subsignals_name=[]
        for k in range(1,predictions.shape[1]+1):
            subsignals_name.append('IMF'+str(k))


    plt.figure(figsize=figsize)
    t=range(full_len-test_len+1,full_len+1)
    for i in range(1,predictions.shape[1]+1):
        plt.subplot(math.ceil(predictions.shape[1]/2.0), 2, i)
        plt.title(subsignals_name[i-1],fontsize=fontsize,loc='left')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if i==predictions.shape[1] or i==predictions.shape[1]-1:
            plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel("flow(" + r"$m^3$" + "/s)", fontsize=fontsize)
        if i==1:
            plt.plot(t,records.iloc[:,i-1], '-', color='blue', label='records',linewidth=1.0)
            plt.plot(t,predictions.iloc[:,i-1], '--', color='red', label='',linewidth=1.0)
        if i==2:
            plt.plot(t,records.iloc[:,i-1], '-', color='blue', label='',linewidth=1.0)
            plt.plot(t,predictions.iloc[:,i-1], '--', color='red', label='predictions',linewidth=1.0)
        plt.plot(t,records.iloc[:,i-1], '-', color='blue', label='',linewidth=1.0)
        plt.plot(t,predictions.iloc[:,i-1], '--', color='red', label='',linewidth=1.0)
        plt.legend(
            loc=3,
            bbox_to_anchor=(0.3,1.02, 1,0.102),
            # bbox_transform=plt.gcf().transFigure,
            ncol=2,
            shadow=False,
            frameon=False,
            fontsize=fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, bottom=0.08, right=0.96,top=0.94, hspace=0.6, wspace=0.3)
    plt.savefig(fig_savepath, transparent=False, format=format, dpi=dpi)
