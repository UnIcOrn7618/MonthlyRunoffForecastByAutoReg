import pandas as pd
import numpy as np
import math
from deprecated import deprecated
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)

from tools.metrics_ import PPTS
from config.globalLog import logger

def dump_train_dev_test_to_csv(
        path,
        train_y=None,
        train_pred=None,
        train_nse=None,
        train_mse=None,
        train_nrmse=None,
        train_mae=None,
        train_mape=None,
        train_ppts=None,
        dev_y=None,
        dev_pred=None,
        dev_nse=None,
        dev_mse=None,
        dev_nrmse=None,
        dev_mae=None,
        dev_mape=None,
        dev_ppts=None,
        test_y=None,
        test_pred=None,
        test_nse=None,
        test_mse=None,
        test_nrmse=None,
        test_mae=None,
        test_mape=None,
        test_ppts=None,
        time_cost=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        train_nse: R square value for training records and predictions, type float.
        train_nrmse: Normalized root mean square error value for training records and predictions, type float.
        train_mae: Mean absolute error value for training records and predictions, type float.
        train_mape: Mean absolute percentage error value for training records and predictions, type float.
        train_ppts: Peak percentage threshold statistic value for training records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        dev_nse: R square value for development records and predictions, type float.
        dev_nrmse: Normalized root mean square error value for development records and predictions, type float.
        dev_mae: Mean absolute error value for development records and predictions, type float.
        dev_mape: Mean absolute percentage error value for development records and predictions, type float.
        dev_ppts: Peak percentage threshold statistic value for development records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        test_nse: R square value for testing records and predictions, type float.
        test_nrmse: Normalized root mean square error value for testing records and predictions, type float.
        test_mae: Mean absolute error value for testing records and predictions, type float.
        test_mape: Mean absolute percentage error value for testing records and predictions, type float.
        test_ppts: Peak percentage threshold statistic value for testing records and predictions, type float.
        time_cost: Time cost for profiling, type float.
    """

    index_train = pd.Index(np.linspace(0, train_y.size-1, train_y.size))
    index_dev = pd.Index(np.linspace(0, dev_y.size-1, dev_y.size))
    index_test = pd.Index(np.linspace(0, test_y.size-1, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    train_nse = pd.DataFrame([train_nse], columns=['train_nse'])['train_nse']
    train_mse = pd.DataFrame([train_mse], columns=['train_mse'])['train_mse']
    train_nrmse = pd.DataFrame([train_nrmse], columns=['train_nrmse'])['train_nrmse']
    train_mae = pd.DataFrame([train_mae], columns=['train_mae'])['train_mae']
    train_mape = pd.DataFrame([train_mape], columns=['train_mape'])['train_mape']
    train_ppts = pd.DataFrame([train_ppts],columns=['train_ppts'])['train_ppts']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y = pd.DataFrame(list(dev_y), index=index_dev, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    dev_nse = pd.DataFrame([dev_nse], columns=['dev_nse'])['dev_nse']
    dev_mse = pd.DataFrame([dev_mse], columns=['dev_mse'])['dev_mse']
    dev_nrmse = pd.DataFrame([dev_nrmse], columns=['dev_nrmse'])['dev_nrmse']
    dev_mae = pd.DataFrame([dev_mae], columns=['dev_mae'])['dev_mae']
    dev_mape = pd.DataFrame([dev_mape], columns=['dev_mape'])['dev_mape']
    dev_ppts = pd.DataFrame([dev_ppts], columns=['dev_ppts'])['dev_ppts']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    test_nse = pd.DataFrame([test_nse], columns=['test_nse'])['test_nse']
    test_mse = pd.DataFrame([test_mse], columns=['test_mse'])['test_mse']
    test_nrmse = pd.DataFrame([test_nrmse], columns=['test_nrmse'])['test_nrmse']
    test_mae = pd.DataFrame([test_mae], columns=['test_mae'])['test_mae']
    test_mape = pd.DataFrame([test_mape], columns=['test_mape'])['test_mape']
    test_ppts = pd.DataFrame([test_ppts], columns=['test_ppts'])['test_ppts']

    time_cost = pd.DataFrame([time_cost], columns=['time_cost'])['time_cost']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                train_nse,
                train_mse,
                train_nrmse,
                train_mae,
                train_mape,
                train_ppts,
                dev_y,
                dev_pred,
                dev_nse,
                dev_mse,
                dev_nrmse,
                dev_mae,
                dev_mape,
                dev_ppts,
                test_y,
                test_pred,
                test_nse,
                test_mse,
                test_nrmse,
                test_mae,
                test_mape,
                test_ppts,
                time_cost,
            ],
            axis=1))
    results.to_csv(path)

def dum_pred_results(path,train_y,train_predictions,dev_y,dev_predictions,test_y,test_predictions,time_cost=None):
    """ 
    Dump real records (labels) and predictions as well as evaluation criteria (metrix R2,RMSE,MAE,MAPE,PPTS,time_cost) to csv.
    Args:
        path: The local disk path to dump data into.
        train_y: records of training set with numpy array type.
        train_predictions: predictions of training set with numpy array type.
        dev_y: records of development set with numpy array type.
        dev_predictions: predictions of development set with numpy array type.
        test_y: records of testing set with numpy array type.
        test_predictions: predictions of testing set with numpy array type.
        time_cost: Time cost for profiling.
    
    Return:
    A csv file
    """
    logger.info('Dump records, predictions and evaluation criteria...')
    logger.info('Compute Nash-Sutcliffe efficiency (NSE)...')
    train_nse = r2_score(train_y, train_predictions)
    dev_nse = r2_score(dev_y, dev_predictions)
    test_nse = r2_score(test_y, test_predictions)
    logger.info('Compute Mean Square Error (MSE)...')
    train_mse = mean_squared_error(y_true=train_y,y_pred=train_predictions)
    dev_mse = mean_squared_error(y_true=dev_y,y_pred=dev_predictions)
    test_mse = mean_squared_error(y_true=test_y,y_pred=test_predictions)
    logger.info('Compute normalized mean square error (NRMSE)...')
    train_nrmse = math.sqrt(mean_squared_error(train_y, train_predictions))/(sum(train_y)/len(train_y))
    dev_nrmse = math.sqrt(mean_squared_error(dev_y, dev_predictions))/(sum(dev_y)/len(dev_y))
    test_nrmse = math.sqrt(mean_squared_error(test_y, test_predictions))/(sum(test_y)/len(test_y))
    logger.info('Compute mean absolute error (MAE)...')
    train_mae = mean_absolute_error(train_y, train_predictions)
    dev_mae = mean_absolute_error(dev_y, dev_predictions)
    test_mae = mean_absolute_error(test_y, test_predictions)
    logger.info('Compute mean absolute percentage error (MAPE)...')
    train_mape=np.mean(np.abs((train_y - train_predictions) / train_y)) * 100
    dev_mape=np.mean(np.abs((dev_y - dev_predictions) / dev_y)) * 100
    test_mape=np.mean(np.abs((test_y - test_predictions) / test_y)) * 100
    logger.info('Compute peak percentage of threshold statistic (PPTS)...')
    train_ppts = PPTS(train_y,train_predictions,5)
    dev_ppts = PPTS(dev_y,dev_predictions,5)
    test_ppts = PPTS(test_y,test_predictions,5)
    logger.info('Dumping the model results.')
    dump_train_dev_test_to_csv(
            path=path,
            train_y=train_y,
            train_pred=train_predictions,
            train_nse=train_nse,
            train_mse = train_mse,
            train_nrmse=train_nrmse,
            train_mae=train_mae,
            train_mape=train_mape,
            train_ppts=train_ppts,
            dev_y=dev_y,
            dev_pred=dev_predictions,
            dev_nse=dev_nse,
            dev_mse = dev_mse,
            dev_nrmse=dev_nrmse,
            dev_mae=dev_mae,
            dev_mape=dev_mape,
            dev_ppts=dev_ppts,
            test_y=test_y,
            test_pred=test_predictions,
            test_nse=test_nse,
            test_mse=test_mse,
            test_nrmse=test_nrmse,
            test_mae=test_mae,
            test_mape=test_mape,
            test_ppts=test_ppts,
            time_cost=time_cost,
            )

