import pandas as pd
import numpy as np
import math
from deprecated import deprecated
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from metrics_ import PPTS

@deprecated("This mnethod is deprecated")
def dump_train_dev_to_excel(
        path,
        train_y=None,
        train_pred=None,
        dev_y=None,
        dev_pred=None,
        test_y=None,
        test_pred=None,
):
    writer = pd.ExcelWriter(path)
    # convert the test_pred numpy array into Dataframe series
    # test_y = pd.DataFrame(test_pred, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, columns=['test_pred'])['test_pred']

    # train_y = pd.DataFrame(train_y, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(train_pred, columns=['train_pred'])['train_pred']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, columns=['dev_pred'])['dev_pred']
    results = pd.DataFrame(
        pd.concat(
            [test_y, test_pred, train_y, train_pred, dev_y, dev_pred], axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()


""" 
y_aa = np.array([1,2,3,4,5,6,7,8,7,9,1,0,11,12,15,14,17])
aa = pd.DataFrame([1,2,3,4,5,6,7,8,9],columns=['aa'])['aa']
y_bb = np.array([11,12,13,14,15,16])
bb = pd.DataFrame(['a','b','c','d','e','f','g','h'],columns=['bb'])['bb']
dump_to_excel('F:/ml_fp_lytm/tf_projects/test/test.xlsx',aa,y_aa,bb,y_bb) 
"""

@deprecated("This mnethod is deprecated")
def dump_train_dev_test_to_excel(
        path,
        train_y=None,
        train_pred=None,
        train_r2=None,
        train_rmse=None,
        train_mae=None,
        train_mape=None,
        train_ppts=None,
        dev_y=None,
        dev_pred=None,
        dev_r2=None,
        dev_rmse=None,
        dev_mae=None,
        dev_mape=None,
        dev_ppts=None,
        test_y=None,
        test_pred=None,
        test_r2=None,
        test_rmse=None,
        test_mae=None,
        test_mape=None,
        test_ppts=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        train_r2: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        dev_r2: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        test_r2: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_train = pd.Index(np.linspace(1, train_y.size, train_y.size))
    index_dev = pd.Index(np.linspace(1, dev_y.size, dev_y.size))
    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    train_r2 = pd.DataFrame([train_r2], columns=['train_r2'])['train_r2']
    train_rmse = pd.DataFrame([train_rmse], columns=['train_rmse'])['train_rmse']
    train_mae = pd.DataFrame([train_mae], columns=['train_mae'])['train_mae']
    train_mape = pd.DataFrame([train_mape], columns=['train_mape'])['train_mape']
    train_ppts = pd.DataFrame([train_ppts],columns=['train_ppts'])['train_ppts']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y = pd.DataFrame(list(dev_y), index=index_dev, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    dev_r2 = pd.DataFrame([dev_r2], columns=['dev_r2'])['dev_r2']
    dev_rmse = pd.DataFrame([dev_rmse], columns=['dev_rmse'])['dev_rmse']
    dev_mae = pd.DataFrame([dev_mae], columns=['dev_mae'])['dev_mae']
    dev_mape = pd.DataFrame([dev_mape], columns=['dev_mape'])['dev_mape']
    dev_ppts = pd.DataFrame([dev_ppts], columns=['dev_ppts'])['dev_ppts']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    test_r2 = pd.DataFrame([test_r2], columns=['test_r2'])['test_r2']
    test_rmse = pd.DataFrame([test_rmse], columns=['test_rmse'])['test_rmse']
    test_mae = pd.DataFrame([test_mae], columns=['test_mae'])['test_mae']
    test_mape = pd.DataFrame([test_mape], columns=['test_mape'])['test_mape']
    test_ppts = pd.DataFrame([test_ppts], columns=['test_ppts'])['test_ppts']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                train_r2,
                train_rmse,
                train_mae,
                train_mape,
                train_ppts,
                dev_y,
                dev_pred,
                dev_r2,
                dev_rmse,
                dev_mae,
                dev_mape,
                dev_ppts,
                test_y,
                test_pred,
                test_r2,
                test_rmse,
                test_mae,
                test_mape,
                test_ppts,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

def dump_train_dev_test_to_csv(
        path,
        train_y=None,
        train_pred=None,
        train_r2=None,
        train_rmse=None,
        train_mae=None,
        train_mape=None,
        train_ppts=None,
        dev_y=None,
        dev_pred=None,
        dev_r2=None,
        dev_rmse=None,
        dev_mae=None,
        dev_mape=None,
        dev_ppts=None,
        test_y=None,
        test_pred=None,
        test_r2=None,
        test_rmse=None,
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
        train_r2: R square value for training records and predictions, type float.
        train_rmse: Root mmean square error value for training records and predictions, type float.
        train_mae: Mean absolute error value for training records and predictions, type float.
        train_mape: Mean absolute percentage error value for training records and predictions, type float.
        train_ppts: Peak percentage threshold statistic value for training records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        dev_r2: R square value for development records and predictions, type float.
        dev_rmse: Root mmean square error value for development records and predictions, type float.
        dev_mae: Mean absolute error value for development records and predictions, type float.
        dev_mape: Mean absolute percentage error value for development records and predictions, type float.
        dev_ppts: Peak percentage threshold statistic value for development records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        test_r2: R square value for testing records and predictions, type float.
        test_rmse: Root mmean square error value for testing records and predictions, type float.
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
    train_r2 = pd.DataFrame([train_r2], columns=['train_r2'])['train_r2']
    train_rmse = pd.DataFrame([train_rmse], columns=['train_rmse'])['train_rmse']
    train_mae = pd.DataFrame([train_mae], columns=['train_mae'])['train_mae']
    train_mape = pd.DataFrame([train_mape], columns=['train_mape'])['train_mape']
    train_ppts = pd.DataFrame([train_ppts],columns=['train_ppts'])['train_ppts']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y = pd.DataFrame(list(dev_y), index=index_dev, columns=['dev_y'])['dev_y']
    dev_pred = pd.DataFrame(dev_pred, index=index_dev, columns=['dev_pred'])['dev_pred']
    dev_r2 = pd.DataFrame([dev_r2], columns=['dev_r2'])['dev_r2']
    dev_rmse = pd.DataFrame([dev_rmse], columns=['dev_rmse'])['dev_rmse']
    dev_mae = pd.DataFrame([dev_mae], columns=['dev_mae'])['dev_mae']
    dev_mape = pd.DataFrame([dev_mape], columns=['dev_mape'])['dev_mape']
    dev_ppts = pd.DataFrame([dev_ppts], columns=['dev_ppts'])['dev_ppts']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    test_r2 = pd.DataFrame([test_r2], columns=['test_r2'])['test_r2']
    test_rmse = pd.DataFrame([test_rmse], columns=['test_rmse'])['test_rmse']
    test_mae = pd.DataFrame([test_mae], columns=['test_mae'])['test_mae']
    test_mape = pd.DataFrame([test_mape], columns=['test_mape'])['test_mape']
    test_ppts = pd.DataFrame([test_ppts], columns=['test_ppts'])['test_ppts']

    time_cost = pd.DataFrame([time_cost], columns=['time_cost'])['time_cost']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                train_r2,
                train_rmse,
                train_mae,
                train_mape,
                train_ppts,
                dev_y,
                dev_pred,
                dev_r2,
                dev_rmse,
                dev_mae,
                dev_mape,
                dev_ppts,
                test_y,
                test_pred,
                test_r2,
                test_rmse,
                test_mae,
                test_mape,
                test_ppts,
                time_cost,
            ],
            axis=1))
    results.to_csv(path)

@deprecated("This mnethod is deprecated")
def dump_train_dev12_test_to_excel(
        path,
        train_y=None,
        train_pred=None,
        train_r2=None,
        train_rmse=None,
        train_mae=None,
        train_mape=None,
        train_ppts=None,
        dev_y1=None,
        dev1_pred=None,
        r2_dev1=None,
        rmse_dev1=None,
        mae_dev1=None,
        mape_dev1=None,
        ppts_dev1=None,
        dev_y2=None,
        dev2_pred=None,
        r2_dev2=None,
        rmse_dev2=None,
        mae_dev2=None,
        mape_dev2=None,
        ppts_dev2=None,
        test_y=None,
        test_pred=None,
        test_r2=None,
        test_rmse=None,
        test_mae=None,
        test_mape=None,
        test_ppts=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        train_r2: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        dev_r2: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        test_r2: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_train = pd.Index(np.linspace(1, train_y.size, train_y.size))
    index_dev1 = pd.Index(np.linspace(1, dev_y1.size, dev_y1.size))
    index_dev2 = pd.Index(np.linspace(1, dev_y2.size, dev_y2.size))
    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    # convert the train_pred numpy array into Dataframe series
    train_y = pd.DataFrame(list(train_y), index=index_train, columns=['train_y'])['train_y']
    train_pred = pd.DataFrame(data=train_pred, index=index_train,columns=['train_pred'])['train_pred']
    train_r2 = pd.DataFrame([train_r2], columns=['train_r2'])['train_r2']
    train_rmse = pd.DataFrame([train_rmse], columns=['train_rmse'])['train_rmse']
    train_mae = pd.DataFrame([train_mae], columns=['train_mae'])['train_mae']
    train_mape = pd.DataFrame([train_mape], columns=['train_mape'])['train_mape']
    train_ppts = pd.DataFrame([train_ppts],columns=['train_ppts'])['train_ppts']

    # dev_y = pd.DataFrame(dev_y, columns=['dev_y'])['dev_y']
    dev_y1 = pd.DataFrame(list(dev_y1), index=index_dev1, columns=['dev_y1'])['dev_y1']
    dev1_pred = pd.DataFrame(dev1_pred, index=index_dev1, columns=['dev1_pred'])['dev1_pred']
    r2_dev1 = pd.DataFrame([r2_dev1], columns=['r2_dev1'])['r2_dev1']
    rmse_dev1 = pd.DataFrame([rmse_dev1], columns=['rmse_dev1'])['rmse_dev1']
    mae_dev1 = pd.DataFrame([mae_dev1], columns=['mae_dev1'])['mae_dev1']
    mape_dev1 = pd.DataFrame([mape_dev1], columns=['mape_dev1'])['mape_dev1']
    ppts_dev1 = pd.DataFrame([ppts_dev1], columns=['ppts_dev1'])['ppts_dev1']

    dev_y2 = pd.DataFrame(list(dev_y2), index=index_dev2, columns=['dev_y2'])['dev_y2']
    dev2_pred = pd.DataFrame(dev2_pred, index=index_dev2, columns=['dev2_pred'])['dev2_pred']
    r2_dev2 = pd.DataFrame([r2_dev2], columns=['r2_dev2'])['r2_dev2']
    rmse_dev2 = pd.DataFrame([rmse_dev2], columns=['rmse_dev2'])['rmse_dev2']
    mae_dev2 = pd.DataFrame([mae_dev2], columns=['mae_dev2'])['mae_dev2']
    mape_dev2 = pd.DataFrame([mape_dev2], columns=['mape_dev2'])['mape_dev2']
    ppts_dev2 = pd.DataFrame([ppts_dev2], columns=['ppts_dev2'])['ppts_dev2']

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    test_r2 = pd.DataFrame([test_r2], columns=['test_r2'])['test_r2']
    test_rmse = pd.DataFrame([test_rmse], columns=['test_rmse'])['test_rmse']
    test_mae = pd.DataFrame([test_mae], columns=['test_mae'])['test_mae']
    test_mape = pd.DataFrame([test_mape], columns=['test_mape'])['test_mape']
    test_ppts = pd.DataFrame([test_ppts], columns=['test_ppts'])['test_ppts']

    results = pd.DataFrame(
        pd.concat(
            [
                train_y,
                train_pred,
                train_r2,
                train_rmse,
                train_mae,
                train_mape,
                train_ppts,
                dev_y1,
                dev1_pred,
                r2_dev1,
                rmse_dev1,
                mae_dev1,
                mape_dev1,
                ppts_dev1,
                dev_y2,
                dev2_pred,
                r2_dev2,
                rmse_dev2,
                mae_dev2,
                mape_dev2,
                ppts_dev2,
                test_y,
                test_pred,
                test_r2,
                test_rmse,
                test_mae,
                test_mape,
                test_ppts,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

@deprecated("This mnethod is deprecated")
def dump_test_to_excel(
        path,
        test_y=None,
        test_pred=None,
        test_r2=None,
        test_rmse=None,
        test_mae=None,
        test_mape=None,
        test_ppts=None,
):
    """ 
    Dump training and developing records and predictions as well as r square to excel.
    Args:
        path: The local disk path to dump data into.
        train_y: train records with Dataframe type.
        train_pred: train predictions with numpy array type.
        train_r2: R square value for train records and predictions, type float.
        dev_y: developing records with Dataframe type.
        dev_pred: developing predictions with numpy array type.
        dev_r2: R square value for developing records and predictions, type float.
        test_y: testing records with Dataframe type.
        test_pred: testing predictions with numpy array type.
        test_r2: R square value for testing records and predictions, type float.
    """
    writer = pd.ExcelWriter(path)

    index_test = pd.Index(np.linspace(1, test_y.size, test_y.size))

    test_y = pd.DataFrame(list(test_y), index=index_test, columns=['test_y'])['test_y']
    test_pred = pd.DataFrame(test_pred, index=index_test, columns=['test_pred'])['test_pred']
    test_r2 = pd.DataFrame([test_r2], columns=['test_r2'])['test_r2']
    test_rmse = pd.DataFrame([test_rmse], columns=['test_rmse'])['test_rmse']
    test_mae = pd.DataFrame([test_mae], columns=['test_mae'])['test_mae']
    test_mape = pd.DataFrame([test_mape], columns=['test_mape'])['test_mape']
    test_ppts = pd.DataFrame([test_ppts], columns=['test_ppts'])['test_ppts']

    results = pd.DataFrame(
        pd.concat(
            [
                test_y,
                test_pred,
                test_r2,
                test_rmse,
                test_mae,
                test_mape,
                test_ppts,
            ],
            axis=1))
    results.to_excel(writer, sheet_name='Sheet1')
    writer.close()

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
    # compute R square
    train_r2 = r2_score(train_y, train_predictions)
    dev_r2 = r2_score(dev_y, dev_predictions)
    test_r2 = r2_score(test_y, test_predictions)
    # compute MSE
    train_rmse = math.sqrt(mean_squared_error(train_y, train_predictions))
    dev_rmse = math.sqrt(mean_squared_error(dev_y, dev_predictions))
    test_rmse = math.sqrt(mean_squared_error(test_y, test_predictions))
    # compute MAE
    train_mae = mean_absolute_error(train_y, train_predictions)
    dev_mae = mean_absolute_error(dev_y, dev_predictions)
    test_mae = mean_absolute_error(test_y, test_predictions)
    # compute MAPE
    train_mape=np.mean(np.abs((train_y - train_predictions) / train_y)) * 100
    dev_mape=np.mean(np.abs((dev_y - dev_predictions) / dev_y)) * 100
    test_mape=np.mean(np.abs((test_y - test_predictions) / test_y)) * 100
    # compute PPTS
    train_ppts = PPTS(train_y,train_predictions,5)
    dev_ppts = PPTS(dev_y,dev_predictions,5)
    test_ppts = PPTS(test_y,test_predictions,5)

    dump_train_dev_test_to_csv(
            path=path,
            train_y=train_y,
            train_pred=train_predictions,
            train_r2=train_r2,
            train_rmse=train_rmse,
            train_mae=train_mae,
            train_mape=train_mape,
            train_ppts=train_ppts,
            dev_y=dev_y,
            dev_pred=dev_predictions,
            dev_r2=dev_r2,
            dev_rmse=dev_rmse,
            dev_mae=dev_mae,
            dev_mape=dev_mape,
            dev_ppts=dev_ppts,
            test_y=test_y,
            test_pred=test_predictions,
            test_r2=test_r2,
            test_rmse=test_rmse,
            test_mae=test_mae,
            test_mape=test_mape,
            test_ppts=test_ppts,
            time_cost=time_cost,
            )

