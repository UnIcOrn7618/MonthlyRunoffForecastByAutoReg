#### import basic external libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.set_cmap("viridis")
import datetime
import time
#### import libs for optimize SVR or GBRT
from sklearn.svm import SVR,NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.externals.joblib import Parallel, delayed
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize,forest_minimize, dummy_minimize
from skopt.plots import plot_convergence,plot_objective,plot_evaluations
from skopt import dump, load
from skopt import Optimizer
from skopt.benchmarks import branin
from functools import partial
from statsmodels.tsa.arima_model import ARIMA
from random import seed
from random import random
# from skopt.callbacks import CheckpointSaver

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

import os
root_path = os.path.dirname(os.path.abspath('_file_'))
import sys
sys.path.append(root_path)
from config.globalLog import logger


# import own coding libs
from tools.plot_utils import plot_convergence_
from tools.plot_utils import plot_evaluations_
from tools.plot_utils import plot_objective_
from tools.plot_utils import plot_rela_pred
from tools.plot_utils import plot_history
from tools.plot_utils import plot_error_distribution
from tools.dump_data import dum_pred_results

ESVR_SPACE = [
        # Penalty parameter `C` of the error term
        Real(0.1, 200, name='C'),   
        # `epsilon` in epsilon-SVR model. It specifies the epsilon-tube
        # within which no penalty is associated in the training loss
        # function with points predicted within a distance epsilon from the actual value.
        Real(10**-6, 10**0, name='epsilon'),    
        # kernel coefficient for 'rbf','poly' and 'sigmoid'
        Real(10**-6, 10**0, name='gamma'),  
    ]
DIMENSION_ESVR = ['C','epsilon','gamma']
DIMENSION_GBRT = ['max depth','learning rate','max features','min samples split','min samples leaf']
EPS_DPI = 2000
TIFF_DPI=1200


def multi_optimizer_esvr(root_path,station,predict_pattern,n_calls=100,cv=6):
    # Set the time series and model parameters
    predictor = 'esvr'
    data_path = root_path + '/'+station+'/data/'+predict_pattern+'/'
    model_path = root_path+'/'+station+'/projects/'+predictor+'/'+predict_pattern+'/multi_optimizer/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = 'nc'+str(n_calls)+'_cv'+str(cv)
    logger.info("Build multiple optimizer epsilon SVR...")
    
    logger.info("Root path:{}".format(root_path))
    logger.info("Station:{}".format(station))
    logger.info("Predict pattern:{}".format(predict_pattern))
    logger.info("Number of calls:{}".format(n_calls))
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    
    if os.path.exists(model_path +model_name+'_optimized_params.csv') :
        optimal_params = pd.read_csv(model_path +model_name+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
    else:
        logger.info('Load learning samples...')
        # Load the training, development and testing samples
        train = pd.read_csv(data_path+'minmax_unsample_train.csv',index_col=False)
        dev = pd.read_csv(data_path+'minmax_unsample_dev.csv',index_col=False)
        test = pd.read_csv(data_path+'minmax_unsample_test.csv',index_col=False)
        train_dev = pd.concat([train,dev],axis=0)
        # shuffle the training samples
        train_dev = train_dev.sample(frac=1)
        train_y = train['Y']
        train_x = train.drop('Y', axis=1)
        dev_y = dev['Y']
        dev_x = dev.drop('Y', axis=1)
        test_y = test['Y']
        test_x = test.drop('Y', axis=1)
        train_dev_y = train_dev['Y']
        train_dev_x = train_dev.drop('Y', axis=1)
        logger.info('Build SVR model and set the evaluation space of Bayesian optimization.')
        reg = SVR(tol=1e-4)
        # Set the space of hyper-parameters for tuning them
        space = ESVR_SPACE
        # Define an objective function of hyper-parameters tuning
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

            def run(minimizer, n_iter=5):
                return [minimizer(objective, space, n_calls=n_calls, random_state=n) 
                        for n in range(n_iter)]
            #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)
            # Random search
            dummy_res = run(dummy_minimize) 
            # Gaussian processes
            gp_res = run(gp_minimize)
            # Random forest
            rf_res = run(partial(forest_minimize, base_estimator="RF"))
            # Extra trees 
            et_res = run(partial(forest_minimize, base_estimator="ET"))


            plot = plot_convergence(("dummy_minimize", dummy_res),
                                ("gp_minimize", gp_res),
                                ("forest_minimize('rf')", rf_res),
                                ("forest_minimize('et)", et_res), 
                                true_minimum=0.397887, yscale="log")

            plot.legend(loc="best", prop={'size': 6}, numpoints=1);
            plt.close('all')
    

def esvr(root_path,station,predict_pattern,optimizer='gp',n_calls=100,cv=6):
    logger.info("Build monoscale epsilon SVR model ...")
    logger.info("Root path:{}".format(root_path))
    logger.info("Station:{}".format(station))
    logger.info("Predict pattern:{}".format(predict_pattern))
    logger.info("Optimizer:{}".format(optimizer))
    logger.info("Number of calls:{}".format(n_calls))
    predictor = 'esvr'
    data_path = root_path + '/'+station+'/data/'+predict_pattern+'/'
    model_path = root_path+'/'+station+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))
    logger.info("Model name:{}".format(model_name))
    
    # Load the training, development and testing samples
    logger.info('Load learning samples...')
    train = pd.read_csv(data_path+'minmax_unsample_train.csv',index_col=False)
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv',index_col=False)
    test = pd.read_csv(data_path+'minmax_unsample_test.csv',index_col=False)
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle the training samples
    train_dev = train_dev.sample(frac=1)
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)

    if os.path.exists(model_path +model_name+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path +model_name+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
            esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()
            norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            train_predictions[train_predictions<0.0]=0.0
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions[dev_predictions<0.0]=0.0
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions[test_predictions<0.0]=0.0
            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
                )
    else:
        reg = SVR(tol=1e-4)
        # Set the space of hyper-parameters for tuning them
        space = ESVR_SPACE
        # Define an objective function of hyper-parameters tuning
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))
        # Tuning the hyper-parameters using Bayesian Optimization based on Gaussion Process
        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_et':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end = time.process_time()
        time_cost = end-start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')

        # Visualizing the results of hyper-parameaters tuning
        plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')

        # Plot the optimal hyperparameters
        logger.info('Best score=%.4f'%res.fun)
        logger.info(""" Best parameters:
         -C = %.8f
         -epsilon = %.8f
         -gamma = %.8f
         """%(res.x[0],res.x[1],res.x[2]))
        logger.info('Time cost:{} seconds'.format(time_cost))

        # Construct the optimal hyperparameters to restore them
        params_dict={
            'C':res.x[0],
            'epsilon':res.x[1],
            'gamma':res.x[2],
            'time_cost':time_cost,
            'n_calls':n_calls,
        }

        # Transform the optimal hyperparameters dict to pandas DataFrame and restore it
        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path +model_name+'_optimized_params.csv')

        # Initialize a SVR with the optimal hyperparameters
        esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
        # Do prediction with the opyimal model
        train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)

        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()

        norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))

        # Renormalized the records and predictions and cap the negative predictions to 0
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        train_predictions[train_predictions<0.0]=0.0
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions[dev_predictions<0.0]=0.0
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions[test_predictions<0.0]=0.0


        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost = time_cost,
            )

        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+model_name+"_test_error.png")
        plt.close('all')
    

def esvr_multi_seed(root_path,station,predict_pattern,optimizer='gp',n_calls=100,cv=6,iterations=10):
    logger.info("Build epsilon SVR with multiple seed...")
    logger.info("Root path:{}".format(root_path))
    logger.info("Station:{}".format(station))
    logger.info("Predict pattern:{}".format(predict_pattern))
    logger.info("Optimizer:{}".format(optimizer))
    logger.info("Number of calls:{}".format(n_calls))
    
    # Set the time series and model parameters
    predictor = 'esvr'
    data_path = root_path + '/'+station+'/data/'+predict_pattern+'/'
    model_path = root_path+'/'+station+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))
        
    for random_state in range(1,iterations+1):
        model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)+'_seed'+str(random_state)
        logger.info('Model Name:{}'.format(model_name))
        # Load the training, development and testing samples
        train = pd.read_csv(data_path+'minmax_unsample_train.csv',index_col=False)
        dev = pd.read_csv(data_path+'minmax_unsample_dev.csv',index_col=False)
        test = pd.read_csv(data_path+'minmax_unsample_test.csv',index_col=False)
        train_dev = pd.concat([train,dev],axis=0)
        # shuffle the training samples
        train_dev = train_dev.sample(frac=1)
        train_y = train['Y']
        train_x = train.drop('Y', axis=1)
        dev_y = dev['Y']
        dev_x = dev.drop('Y', axis=1)
        test_y = test['Y']
        test_x = test.drop('Y', axis=1)
        train_dev_y = train_dev['Y']
        train_dev_x = train_dev.drop('Y', axis=1)
        logger.info("Optimized params:{}".format(model_path +model_name+'_optimized_params.csv'))

        if os.path.exists(model_path +model_name+'_optimized_params.csv'):
            optimal_params = pd.read_csv(model_path +model_name+'_optimized_params.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                logger.info("The n_calls="+str(n_calls)+" was already tuned")
                esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
                # Do prediction with the opyimal model
                train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
                dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
                test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
                train_y=(train_y.values).flatten()
                dev_y=(dev_y.values).flatten()
                test_y=(test_y.values).flatten()
                norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
                sMin = norm_id['series_min'][norm_id.shape[0]-1]
                sMax = norm_id['series_max'][norm_id.shape[0]-1]
                logger.debug('Series Min:\n {}'.format(sMin))
                logger.debug('Series Max:\n {}'.format(sMax))
                train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
                dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
                test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
                train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
                train_predictions[train_predictions<0.0]=0.0
                dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
                dev_predictions[dev_predictions<0.0]=0.0
                test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
                test_predictions[test_predictions<0.0]=0.0
                dum_pred_results(
                    path = model_path+model_name+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                    )
        else:
            reg = SVR(tol=1e-4)
            # Set the space of hyper-parameters for tuning them
            space = ESVR_SPACE
            # Define an objective function of hyper-parameters tuning
            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))
            # Tuning the hyper-parameters using Bayesian Optimization based on Gaussion Process
            start = time.process_time()
            if optimizer=='gp':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='fr_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='fr_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='dm':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end = time.process_time()
            time_cost = end-start
            dump(res,model_path+model_name+'_result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+model_name+'_result_seed'+str(random_state)+'.pkl')
            # Visualizing the results of hyper-parameaters tuning
            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')
            # Plot the optimal hyperparameters
            logger.info('Best score=%.4f'%res.fun)
            logger.info(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))
            logger.inf('Time cost:{} seconds'.format(time_cost))
            # Construct the optimal hyperparameters to restore them
            params_dict={
                'C':res.x[0],
                'epsilon':res.x[1],
                'gamma':res.x[2],
                'time_cost':time_cost,
                'n_calls':n_calls,
            }
            # Transform the optimal hyperparameters dict to pandas DataFrame and restore it
            params_df = pd.DataFrame(params_dict,index=[0])
            params_df.to_csv(model_path +model_name+'_optimized_params.csv')
            # Initialize a SVR with the optimal hyperparameters
            esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()
            norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))
            # Renormalized the records and predictions and cap the negative predictions to 0
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            train_predictions[train_predictions<0.0]=0.0
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions[dev_predictions<0.0]=0.0
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions[test_predictions<0.0]=0.0
            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = time_cost,
                )
            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +model_name + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +model_name + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+model_name+"_test_error.png")
    plt.close('all')
        

def one_step_esvr(root_path,station,decomposer,predict_pattern,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6):
    # Set project parameters
    logger.info('Build one-step epsilon SVR model...')
    
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Optimizer:{}'.format(optimizer))
    logger.info('Monther wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of calls:{}'.format(n_calls))
    predictor = 'esvr'
    signals = station+'_'+decomposer
    if decomposer == 'dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    # load data
    train = pd.read_csv(data_path+'minmax_unsample_train.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test.csv')
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle
    train_dev = train_dev.sample(frac=1)
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)
    if os.path.exists(model_path + model_name+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path + model_name+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
            esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()
            norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))
            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            train_predictions[train_predictions<0.0]=0.0
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions[dev_predictions<0.0]=0.0
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions[test_predictions<0.0]=0.0
            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
            )
    else:
        reg = SVR(tol=1e-4)
        space = ESVR_SPACE
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

        #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)

        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end = time.process_time()
        time_cost = end -start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')

        plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')

        logger.info('Best score=%.4f'%res.fun)
        logger.info(""" Best parameters:
         -C = %.8f
         -epsilon = %.8f
         -gamma = %.8f
         """%(res.x[0],res.x[1],res.x[2]))

        logger.info('Time cost:{}'.format(time_cost))
        params_dict={
            'C':res.x[0],
            'epsilon':res.x[1],
            'gamma':res.x[2],
            'time_cost':(time_cost),
            'n_calls':n_calls,
        }

        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path + model_name+'_optimized_params.csv')

        esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
        # Do prediction with the opyimal model
        train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)

        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()

        norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))

        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        train_predictions[train_predictions<0.0]=0.0
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions[dev_predictions<0.0]=0.0
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions[test_predictions<0.0]=0.0

        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)

        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_error.png")
        plt.close('all')
    

def one_step_esvr_multi_seed(root_path,station,decomposer,predict_pattern,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6,iterations=10):
    logger.info('Build one-step epsilon SVR model with multiple seed...')
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Optimizer:{}'.format(optimizer))
    logger.info('Monther wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of calls:{}'.format(n_calls))
    logger.info('Seed iterations:{}'.format(iterations))
    predictor = 'esvr'
    signals = station+'_'+decomposer
    if decomposer == 'dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    for random_state in range(1,iterations+1):
        model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)+'_seed'+str(random_state)
        logger.info('Model Name:{}'.format(model_name))
        # load data
        train = pd.read_csv(data_path+'minmax_unsample_train.csv')
        dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
        test = pd.read_csv(data_path+'minmax_unsample_test.csv')
        train_dev = pd.concat([train,dev],axis=0)
        # shuffle
        train_dev = train_dev.sample(frac=1)
        train_y = train['Y']
        train_x = train.drop('Y', axis=1)
        dev_y = dev['Y']
        dev_x = dev.drop('Y', axis=1)
        test_y = test['Y']
        test_x = test.drop('Y', axis=1)
        train_dev_y = train_dev['Y']
        train_dev_x = train_dev.drop('Y', axis=1)
        logger.info("Optimized params:{}".format(model_path +model_name+'_optimized_params.csv'))
        if os.path.exists(model_path + model_name+'_optimized_params.csv'):
            optimal_params = pd.read_csv(model_path + model_name+'_optimized_params.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                logger.info("The n_calls="+str(n_calls)+" was already tuned")
                esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
                # Do prediction with the opyimal model
                train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
                dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
                test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
                train_y=(train_y.values).flatten()
                dev_y=(dev_y.values).flatten()
                test_y=(test_y.values).flatten()
                norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
                sMin = norm_id['series_min'][norm_id.shape[0]-1]
                sMax = norm_id['series_max'][norm_id.shape[0]-1]
                logger.debug('Series Min:\n {}'.format(sMin))
                logger.debug('Series Max:\n {}'.format(sMax))
                train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
                dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
                test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
                train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
                train_predictions[train_predictions<0.0]=0.0
                dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
                dev_predictions[dev_predictions<0.0]=0.0
                test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
                test_predictions[test_predictions<0.0]=0.0
                dum_pred_results(
                    path = model_path+model_name+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                )
        else:
            reg = SVR(tol=1e-4)
            space = ESVR_SPACE
            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))
            #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)
            start = time.process_time()
            if optimizer=='gp':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='fr_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='fr_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=random_state,verbose=True,n_jobs=-1)
            elif optimizer=='dm':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end = time.process_time()
            time_cost = end -start
            dump(res,model_path+model_name+'_result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+model_name+'_result_seed'+str(random_state)+'.pkl')

            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')

            logger.info('Best score=%.4f'%res.fun)
            logger.info(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))

            logger.info('Time cost:{}'.format(time_cost))
            params_dict={
                'C':res.x[0],
                'epsilon':res.x[1],
                'gamma':res.x[2],
                'time_cost':(time_cost),
                'n_calls':n_calls,
            }

            params_df = pd.DataFrame(params_dict,index=[0])
            params_df.to_csv(model_path + model_name+'_optimized_params.csv')

            esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)

            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()

            norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))

            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            train_predictions[train_predictions<0.0]=0.0
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions[dev_predictions<0.0]=0.0
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions[test_predictions<0.0]=0.0
            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost=time_cost)
            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +model_name + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +model_name + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_error.png")
    plt.close('all')
            

        

def multi_step_esvr(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6):
    logger.info('Build multi-step epsilon SVR model...')
    
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Lags:{}'.format(lags))
    logger.info('Model index:{}'.format(model_id))
    logger.info('Optimizer:{}'.format(optimizer))
    logger.info('Mother wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of calls:{}'.format(n_calls))
    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    predictor = 'esvr'
    signals = station+'_'+decomposer
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)+'_s'+str(model_id)
    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))
    train = pd.read_csv(data_path+'minmax_unsample_train_s'+str(model_id)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_s'+str(model_id)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_s'+str(model_id)+'.csv')
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle
    train_dev = train_dev.sample(frac=1)
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)
    logger.info("Optimized params:{}".format(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv'))
    if os.path.exists(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv'):
        optimal_params = pd.read_csv(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
            esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()
            norm_id = pd.read_csv(data_path + 'norm_unsample_id_s'+str(model_id)+'.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))
            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
            )
    else:
        reg = SVR(tol=1e-4)
        space = ESVR_SPACE
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))
    
        #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)
    
        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end=time.process_time()
        time_cost = end -start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')
    
        plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')
        
        
        logger.info('Best score=%.4f'%res.fun)
        logger.info(""" Best parameters:
         -C = %.8f
         -epsilon = %.8f
         -gamma = %.8f
         """%(res.x[0],res.x[1],res.x[2]))
    
        logger.info('Time cost:{}'.format(time_cost))
        params_dict={
            'C':res.x[0],
            'epsilon':res.x[1],
            'gamma':res.x[2],
            'time_cost':(time_cost),
            'n_calls':n_calls,
        }
    
        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv')
    
        esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
        # Do prediction with the opyimal model
        train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
    
        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()
    
        norm_id = pd.read_csv(data_path + 'norm_unsample_id_s' + str(model_id) + '.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))
    
        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
    
    
        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)
    
        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_error.png",)
    plt.close('all')
        


def multi_step_esvr_multi_seed(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6,iterations=10):
    logger.info('Roo path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Lags:{}'.format(lags))
    logger.info('Model index:{}'.format(model_id))
    logger.info('Optimizer:{}'.format(optimizer))

    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
   
    predictor = 'esvr'
    signals = station+'_'+decomposer
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for random_state in range(1,iterations+1):
        model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)+'_s'+str(model_id)+'_seed'+str(random_state)
        logger.info("Data Path:{}".format(data_path))
        logger.info("Model Path:{}".format(model_path))

        train = pd.read_csv(data_path+'minmax_unsample_train_s'+str(model_id)+'.csv')
        dev = pd.read_csv(data_path+'minmax_unsample_dev_s'+str(model_id)+'.csv')
        test = pd.read_csv(data_path+'minmax_unsample_test_s'+str(model_id)+'.csv')
        train_dev = pd.concat([train,dev],axis=0)
        # shuffle
        train_dev = train_dev.sample(frac=1)
        train_y = train['Y']
        train_x = train.drop('Y', axis=1)
        dev_y = dev['Y']
        dev_x = dev.drop('Y', axis=1)
        test_y = test['Y']
        test_x = test.drop('Y', axis=1)
        train_dev_y = train_dev['Y']
        train_dev_x = train_dev.drop('Y', axis=1)
        logger.info("Optimized params:{}".format(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv'))
        if os.path.exists(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv'):
            optimal_params = pd.read_csv(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                logger.info("The n_calls="+str(n_calls)+" was already tuned")
                esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
                # Do prediction with the opyimal model
                train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
                dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
                test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
                train_y=(train_y.values).flatten()
                dev_y=(dev_y.values).flatten()
                test_y=(test_y.values).flatten()
                norm_id = pd.read_csv(data_path + 'norm_unsample_id_s'+str(model_id)+'.csv')
                sMin = norm_id['series_min'][norm_id.shape[0]-1]
                sMax = norm_id['series_max'][norm_id.shape[0]-1]
                logger.debug('Series Min:\n {}'.format(sMin))
                logger.debug('Series Max:\n {}'.format(sMax))
                # Renormalized the records and predictions
                train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
                dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
                test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
                train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
                dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
                test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
                dum_pred_results(
                    path = model_path+model_name+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                )
        else:
            reg = SVR(tol=1e-4)
            space = ESVR_SPACE
            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

            #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)

            start = time.process_time()
            if optimizer=='gp':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
            elif optimizer=='fr_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
            elif optimizer=='fr_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
            elif optimizer=='dm':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end=time.process_time()
            time_cost = end -start
            dump(res,model_path+model_name+'_result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+model_name+'_result_seed'+str(random_state)+'.pkl')
            
            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+model_name+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')


            logger.info('Best score=%.4f'%res.fun)
            logger.info(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))

            logger.info('Time cost:{}'.format(time_cost))
            params_dict={
                'C':res.x[0],
                'epsilon':res.x[1],
                'gamma':res.x[2],
                'time_cost':(time_cost),
                'n_calls':n_calls,
            }

            params_df = pd.DataFrame(params_dict,index=[0])
            params_df.to_csv(model_path + model_name +'_optimized_params_s' + str(model_id) +'.csv')

            esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)

            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()

            norm_id = pd.read_csv(data_path + 'norm_unsample_id_s' + str(model_id) + '.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            logger.debug('Series Min:\n {}'.format(sMin))
            logger.debug('Series Max:\n {}'.format(sMax))

            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin


            dum_pred_results(
                path = model_path+model_name+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost=time_cost)

            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_error.png",)
    plt.close('all')          


def gbrt(root_path,station,predict_pattern,optimizer='gp',n_calls=100,cv=6):
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Optimizer:{}'.format(optimizer))
    logger.info('Number of calls:{}'.format(n_calls))
    predictor = 'gbrt'
    data_path = root_path + '/'+station+'/data/'
    model_path = root_path+'/'+station+'/projects/'+predictor+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)

    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    # load data
    train = pd.read_csv(data_path+'minmax_unsample_train.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test.csv')
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle
    train_dev = train_dev.sample(frac=1)
    assert train.shape[1]==dev.shape[1]==test.shape[1]==train_dev.shape[1]
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)
    
    if os.path.exists(model_path +model_name+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path +model_name+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
    else:
        # Get the feature num
        n_features = train_dev_x.shape[1]
        reg = GradientBoostingRegressor(n_estimators=100,random_state=0)

        # The list hyper-parameters we want
        space = [
            Integer(1,25,name='max_depth'),
            Real(10**-5,10**0,'log-uniform',name='learning_rate'),
            Integer(1,n_features,name='max_features'),
            Integer(2,100,name='min_samples_split'),
            Integer(1,100,name='min_samples_leaf'),
        ]

        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

        #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)

        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end=time.process_time()
        time_cost = end-start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')

        plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')


        logger.info('Best score=%.4f'%res.fun)
        logger.info("""Best parameters:
        - max_depth=%d
        - learning_rate=%.6f
        - max_features=%d
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                    res.x[4]))
        # end=datetime.datetime.now()
        logger.info('Time cost:{}'.format(time_cost))

        params_dict={
            'max_depth':res.x[0],
            'learning_rate':res.x[1],
            'max_features':res.x[2],
            'min_samples_split':res.x[3],
            'min_samples_leaf':res.x[4],
            'time_cost':time_cost,
            'n_calls':n_calls,
        }

        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path +model_name+'_optimized_params.csv')

        GBR = GradientBoostingRegressor(
            max_depth=res.x[0],
            learning_rate=res.x[1],
            max_features=res.x[2],
            min_samples_split=res.x[3],
            min_samples_leaf=res.x[4])

        # Do prediction with the opyimal model
        train_predictions = GBR.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = GBR.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = GBR.fit(train_dev_x,train_dev_y).predict(test_x)

        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()

        norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))

        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        train_predictions[train_predictions<0.0]=0.0
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions[dev_predictions<0.0]=0.0
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions[test_predictions<0.0]=0.0


        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)

        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +model_name + "_test_error.png")
        plt.close('all')

def one_step_gbrt(root_path,station,decomposer,predict_pattern,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6):
    logger.info('Roo path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Optimizer:{}'.format(optimizer))

    predictor = 'gbrt'
    signals = station+'_'+decomposer
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = optimizer+'_nc'+str(n_calls)+'_cv'+str(cv)

    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    # load data
    train = pd.read_csv(data_path+'minmax_unsample_train.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test.csv')
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle
    train_dev = train_dev.sample(frac=1)
    
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)
    
    if os.path.exists(model_path + model_name+ '_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path + model_name+ '_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
    else:
        n_features = train_dev_x.shape[1]
        reg = GradientBoostingRegressor(n_estimators=100,random_state=0)

        # The list hyper-parameters we want
        space = [
            Integer(1,25,name='max_depth'),
            Real(10**-5,10**0,'log-uniform',name='learning_rate'),
            Integer(1,n_features,name='max_features'),
            Integer(2,100,name='min_samples_split'),
            Integer(1,100,name='min_samples_leaf'),
        ]

        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

        #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)

        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end=time.process_time()
        time_cost = end - start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')

        plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')


        logger.info('Best score=%.4f'%res.fun)
        logger.info("""Best parameters:
        - max_depth=%d
        - learning_rate=%.6f
        - max_features=%d
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                    res.x[4]))
        # end=datetime.datetime.now()
        logger.info('Time cost:{}'.format(time_cost))

        params_dict={
            'max_depth':res.x[0],
            'learning_rate':res.x[1],
            'max_features':res.x[2],
            'min_samples_split':res.x[3],
            'min_samples_leaf':res.x[4],
            'time_cost':(time_cost),
            'n_calls':n_calls,
        }

        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path + model_name+ '_optimized_params.csv')

        GBR = GradientBoostingRegressor(
            max_depth=res.x[0],
            learning_rate=res.x[1],
            max_features=res.x[2],
            min_samples_split=res.x[3],
            min_samples_leaf=res.x[4])

        # Do prediction with the opyimal model
        train_predictions = GBR.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = GBR.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = GBR.fit(train_dev_x,train_dev_y).predict(test_x)

        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()

        norm_id = pd.read_csv(data_path + 'norm_unsample_id.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))

        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        train_predictions[train_predictions<0.0]=0.0
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions[dev_predictions<0.0]=0.0
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions[test_predictions<0.0]=0.0


        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)

        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_error.png",)
        plt.close('all')


def multi_step_gbrt(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp',wavelet_level='db10-2',n_calls=100,cv=6):
    logger.info('Roo path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Lags:{}'.format(lags))
    logger.info('Model index:{}'.format(model_id))
    logger.info('Optimizer:{}'.format(optimizer))
    logger.info('Monther wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of calls:{}'.format(n_calls))

    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    # Set project parameters
    predictor = 'gbrt'
    signals = station+'_'+decomposer
    # Set the mode id:
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = station+'_'+decomposer+'_'+predictor+'_'+predict_pattern+'_s'+str(model_id)

    logger.info("Data Path:{}".format(data_path))
    logger.info("Model Path:{}".format(model_path))

    # load data
    train = pd.read_csv(data_path+'minmax_unsample_train_s'+str(model_id)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_s'+str(model_id)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_s'+str(model_id)+'.csv')
    train_dev = pd.concat([train,dev],axis=0)
    # shuffle
    train_dev = train_dev.sample(frac=1)
    
    train_y = train['Y']
    train_x = train.drop('Y', axis=1)
    dev_y = dev['Y']
    dev_x = dev.drop('Y', axis=1)
    test_y = test['Y']
    test_x = test.drop('Y', axis=1)
    train_dev_y = train_dev['Y']
    train_dev_x = train_dev.drop('Y', axis=1)
    
    if os.path.exists(model_path + model_name+'_optimized_params_s' + str(model_id) +'.csv'):
        optimal_params = pd.read_csv(model_path + model_name+'_optimized_params_s' + str(model_id) +'.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            logger.info("The n_calls="+str(n_calls)+" was already tuned")
    else:
        n_features = train_dev_x.shape[1]
        reg = GradientBoostingRegressor(n_estimators=100,random_state=0)

        # The list hyper-parameters we want
        space = [
            Integer(1,25,name='max_depth'),
            Real(10**-5,10**0,'log-uniform',name='learning_rate'),
            Integer(1,n_features,name='max_features'),
            Integer(2,100,name='min_samples_split'),
            Integer(1,100,name='min_samples_leaf'),
        ]

        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=cv,n_jobs=-1,scoring='neg_mean_squared_error'))

        #checkpoint_saver = CheckpointSaver(model_path+model_name+'/checkpoint.pkl',compress=9)

        start = time.process_time()
        if optimizer=='gp':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='fr_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True,n_jobs=-1)
        elif optimizer=='dm':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end=time.process_time()
        time_cost = end -start
        dump(res,model_path+model_name+'_result.pkl',store_objective=False)
        returned_results = load(model_path+model_name+'_result.pkl')

        plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+model_name+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+model_name+'_convergence.png')


        logger.info('Best score=%.4f'%res.fun)
        logger.info("""Best parameters:
        - max_depth=%d
        - learning_rate=%.6f
        - max_features=%d
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                    res.x[4]))
        # end=datetime.datetime.now()
        logger.info('Time cost:{}'.format(time_cost))

        params_dict={
            'max_depth':res.x[0],
            'learning_rate':res.x[1],
            'max_features':res.x[2],
            'min_samples_split':res.x[3],
            'min_samples_leaf':res.x[4],
            'time_cost':(time_cost),
            'n_calls':n_calls,
        }

        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path + model_name+'_optimized_params_s' + str(model_id) +'.csv')

        GBR = GradientBoostingRegressor(
            max_depth=res.x[0],
            learning_rate=res.x[1],
            max_features=res.x[2],
            min_samples_split=res.x[3],
            min_samples_leaf=res.x[4])

        # Do prediction with the opyimal model
        train_predictions = GBR.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = GBR.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = GBR.fit(train_dev_x,train_dev_y).predict(test_x)

        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()

        norm_id = pd.read_csv(data_path + 'norm_unsample_id_s' + str(model_id) + '.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        logger.debug('Series Min:\n {}'.format(sMin))
        logger.debug('Series Max:\n {}'.format(sMax))

        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin


        dum_pred_results(
            path = model_path+model_name+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)

        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + model_name + "_test_error.png")
        plt.close('all')

def lstm(root_path,station,predict_pattern,seed,
    n_epochs=1000,
    batch_size=128,
    learn_rate=0.007,
    decay_rate=0.0,
    n_hidden_layers=1,
    hidden_units=[8],
    dropout_rates=[0.0],
    early_stop=True,
    retrain=False,
    warm_up=False,
    initial_epoch=None,
    ):
    logger.info('Build monoscale LSTM model...')
    logger.info('Model informattion:')
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Seed:{}'.format(seed))
    logger.info('Number of epochs:{}'.format(n_epochs))
    logger.info('Batch size:{}'.format(batch_size))
    logger.info('Learning rate:{}'.format(learn_rate))
    logger.info('Decay rate of learning rate:{}'.format(decay_rate))
    logger.info('Number of hidden layers:{}'.format(n_hidden_layers))
    logger.info('Number of hidden units:{}'.format(hidden_units))
    logger.info('Dropout rates:{}'.format(dropout_rates))
    logger.info('Early stoping:{}'.format(early_stop))
    logger.info('Retrain model:{}'.format(retrain))
    logger.info('Warm up:{}'.format(warm_up))
    logger.info('Initial epoch of warm up:{}'.format(initial_epoch))

    predictor = 'lstm'
    data_path = root_path + '/'+station+'/data/'+predict_pattern+'/'
    model_path = root_path+'/'+station+'/projects/'+predictor+'/'+predict_pattern+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info('Data path:{}'.format(data_path))
    logger.info('Model path:{}'.format(model_path))

    # 1.Import the sampled normalized data set from disk
    logger.info('Load learning samples...')
    train = pd.read_csv(data_path+'minmax_unsample_train.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test.csv')
    train_x = train
    train_y = train.pop('Y')
    train_y = train_y.as_matrix()
    dev_x = dev
    dev_y = dev.pop('Y')
    dev_y = dev_y.as_matrix()
    test_x = test
    test_y = test.pop('Y')
    test_y = test_y.as_matrix()
    # reshape the input features for LSTM
    train_x = (train_x.values).reshape(train_x.shape[0],1,train_x.shape[1])
    dev_x = (dev_x.values).reshape(dev_x.shape[0],1,dev_x.shape[1])
    test_x = (test_x.values).reshape(test_x.shape[0],1,test_x.shape[1])

    model_name = 'LSTM-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(n_epochs)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    
    
    def build_model():
        logger.info('Define LSTM model...')
        if n_hidden_layers==2:
            model = keras.Sequential(
            [
                layers.LSTM(hidden_units[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                layers.LSTM(hidden_units[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(dropout_rates[1], noise_shape=None, seed=seed),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(hidden_units[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(learn_rate,decay=decay_rate)
        model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error','mean_squared_error'])
        return model
    logger.info('Set model parameters restore path...')
    cp_path = model_path+model_name+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+model_name+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger.info('checkpoint dir:{}'.format(checkpoint_dir))
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,mode='min',save_weights_only=True,verbose=1)
    # cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,period=5,verbose=1)
    # if not RESUME_TRAINING:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(checkpoint_dir, ignore_errors=True)
    # else:
    #     print("Resuming training...")
    # initialize a new model
    model = build_model()
    model.summary() #print a simple description for the model
    """
    # Evaluate before training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    # Try the model with initial weights and biases
    example_batch = train_x[:10]
    example_result = model.predict(example_batch)
    print(example_result)
    """
    # 3.Train the model
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
    files = os.listdir(checkpoint_dir)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',min_lr=0.00001,factor=0.2, verbose=1,patience=10, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=100,restore_best_weights=True)

    warm_dir = 'LSTM-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(initial_epoch)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    logger.info("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    logger.info('Train the LSTM model ...')
    if  retrain: # Retraining the LSTM model
        logger.info('retrain the model')
        if early_stop:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,
            callbacks=[
                cp_callback,
                early_stopping,
            ])
            end = time.process_time()
            time_cost = end-start
        else:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[cp_callback])
            end =time.process_time()
            time_cost = end-start
        # # Visualize the model's training progress using the stats stored in the history object
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
    elif len(files)==0: # The current model has not been trained
        if os.path.exists(model_path+warm_dir) and warm_up: # Training the model using the trained weights and biases as initialized parameters
            logger.info('WARM UP FROM EPOCH '+str(initial_epoch)) # Warm up from the last epoch of the target model
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if early_stop:
                start=time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=initial_epoch,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=initial_epoch,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
        else: # Training entirely new model
            logger.info('new train')
            if early_stop:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end -start
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=n_epochs,batch_size=batch_size ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
    else:
        logger.info('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+model_name+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)
        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    logger.info('Predict the training, development and testing samples...')
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id.csv')
    sMax = norm['series_max'][norm.shape[0]-1]
    sMin = norm['series_min'][norm.shape[0]-1]
    logger.debug('Series min:{}'.format(sMin))
    logger.debug('Series max:{}'.format(sMax))

    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1,sMax - sMin) / 2 + sMin
    train_predictions[train_predictions<0.0]=0.0
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1,sMax - sMin) / 2 + sMin
    dev_predictions[dev_predictions<0.0]=0.0
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1,sMax - sMin) / 2 + sMin
    test_predictions[test_predictions<0.0]=0.0
    logger.info('Dump the prediction results...')
    dum_pred_results(
        path = model_path+model_name+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost,
        )
    logger.info('Plot the prediction results...')
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+model_name+'-TEST-ERROR-DSTRI.png')
    plt.close('all')

def one_step_lstm(
        root_path,station,decomposer,predict_pattern,seed,
        wavelet_level='db10-2',
        n_epochs=1000,
        batch_size=128,
        learn_rate=0.007,
        decay_rate=0.0,
        n_hidden_layers=1,
        hidden_units=[8],
        dropout_rates=[0.0],
        early_stop=True,
        retrain=False,
        warm_up=False,
        initial_epoch=None,
    ):
    logger.info('Build one-step LSTM model...')
    logger.info('Model informattion:')
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Seed:{}'.format(seed))
    logger.info('Monther wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of epochs:{}'.format(n_epochs))
    logger.info('Batch size:{}'.format(batch_size))
    logger.info('Learning rate:{}'.format(learn_rate))
    logger.info('Decay rate of learning rate:{}'.format(decay_rate))
    logger.info('Number of hidden layers:{}'.format(n_hidden_layers))
    logger.info('Number of hidden units:{}'.format(hidden_units))
    logger.info('Dropout rates:{}'.format(dropout_rates))
    logger.info('Early stoping:{}'.format(early_stop))
    logger.info('Retrain model:{}'.format(retrain))
    logger.info('Warm up:{}'.format(warm_up))
    logger.info('Initial epoch of warm up:{}'.format(initial_epoch))

    # Set project parameters
    predictor = 'lstm'
    predict_pattern = predict_pattern # hindcast or forecast
    signals = station+'_'+decomposer
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/history/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info('Data path:{}'.format(data_path))
    logger.info('Model path:{}'.format(model_path))
    ######################################################
   
    logger.info('Load learning samples...')
    # 1.Import the sampled normalized data set from disk
    train = pd.read_csv(data_path+'minmax_unsample_train.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test.csv')
    # Split features from labels
    train_x = train
    train_y = train.pop('Y')
    train_y = train_y.as_matrix()
    dev_x = dev
    dev_y = dev.pop('Y')
    dev_y = dev_y.as_matrix()
    test_x = test
    test_y = test.pop('Y')
    test_y = test_y.as_matrix()
    # reshape the input features for LSTM
    train_x = (train_x.values).reshape(train_x.shape[0],1,train_x.shape[1])
    dev_x = (dev_x.values).reshape(dev_x.shape[0],1,dev_x.shape[1])
    test_x = (test_x.values).reshape(test_x.shape[0],1,test_x.shape[1])

    # 2.Build LSTM model with keras  
    model_name = 'LSTM-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(n_epochs)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    # RESUME_TRAINING = True
    def build_model():
        logger.info('Build LSTM model...')
        if n_hidden_layers==2:
            model = keras.Sequential(
            [
                layers.LSTM(hidden_units[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                layers.LSTM(hidden_units[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(dropout_rates[1], noise_shape=None, seed=seed),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(hidden_units[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(learn_rate,decay=decay_rate)
        model.compile(loss='mean_squared_error',optimizer=optimizer,
                        metrics=['mean_absolute_error','mean_squared_error'])
        return model
    logger.info('Set model parameters restore path...')
    # set model's parameters restore path
    cp_path = model_path+model_name+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+model_name+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger.info('checkpoint dir:{}'.format(checkpoint_dir))
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,mode='min',save_weights_only=True,verbose=1)
    # cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,period=5,verbose=1)
    # if not RESUME_TRAINING:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(checkpoint_dir, ignore_errors=True)
    # else:
    #     print("Resuming training...")
    # initialize a new model
    model = build_model()
    model.summary() #print a simple description for the model
    """
    # Evaluate before training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    # Try the model with initial weights and biases
    example_batch = train_x[:10]
    example_result = model.predict(example_batch)
    print(example_result)
    """
    # 3.Train the model
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
    files = os.listdir(checkpoint_dir)

    from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',min_lr=0.00001,factor=0.2, verbose=1,patience=10, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=200,restore_best_weights=True)


    warm_dir = 'LSTM-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(initial_epoch)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    logger.info("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    # Training models
    logger.info('Train the LSTM model...')
    if  retrain: # Retraining the LSTM model
        print('retrain the model')
        if early_stop:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,
            batch_size=batch_size ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
                early_stopping,
            ])
            end = time.process_time()
            time_cost = end-start
        else:
            start=time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,
            batch_size=batch_size ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
            ])
            end = time.process_time()
            time_cost = end - start
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,
        model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
        model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')

    elif len(files)==0:# The current model has not been trained
        # Training the model using the trained weights and biases as initialized parameters
        if os.path.exists(model_path+warm_dir) and warm_up:
            # Warm up from the last epoch of the target model
            logger.info('WARM UP FROM EPOCH '+str(initial_epoch))
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if early_stop:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                initial_epoch=initial_epoch,
                epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end-start+prev_time_cost
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                initial_epoch=initial_epoch,
                epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
            model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
        else:
            print('new train')
            if early_stop:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end - start
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[cp_callback,])
                end = time.process_time()
                time_cost = end - start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
            model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
    else:
        logger.info('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+model_name+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)

        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    # 4. Predict the model
    # load the unsample data
    logger.info('Predict the training, development and testing samples...')
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id.csv')
    sMax = norm['series_max'][norm.shape[0]-1]
    sMin = norm['series_min'][norm.shape[0]-1]
    logger.debug('Series min:{}'.format(sMin))
    logger.debug('Series max:{}'.format(sMax))

    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
    train_predictions[train_predictions<0.0]=0.0
    dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
    dev_predictions[dev_predictions<0.0]=0.0
    test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
    test_predictions[test_predictions<0.0]=0.0
    logger.info('Dump prediction results...')
    dum_pred_results(
        path = model_path+model_name+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost = time_cost)
    logger.info('Plot the prediction results...')
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+model_name+'-TEST-ERROR-DSTRI.png')
    plt.close('all')


def multi_step_lstm(
    root_path,station,decomposer,predict_pattern,lags,model_id,seed,
    wavelet_level='db10-2',
    n_epochs=1000,
    batch_size=128,
    learn_rate=0.007,
    decay_rate=0.0,
    n_hidden_layers=1,
    hidden_units=[8],
    dropout_rates=[0.0],
    early_stop=True,
    retrain=False,
    warm_up=False,
    initial_epoch=None,
):
    logger.info('Build multi-step LSTM model...')
    logger.info('Model informattion:')
    logger.info('Root path:{}'.format(root_path))
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Predict pattern:{}'.format(predict_pattern))
    logger.info('Lags:{}'.format(lags))
    logger.info('Model index:{}'.format(model_id))
    logger.info('Seed:{}'.format(seed))
    logger.info('Monther wavelet and decomposition level of WA:{}'.format(wavelet_level))
    logger.info('Number of epochs:{}'.format(n_epochs))
    logger.info('Batch size:{}'.format(batch_size))
    logger.info('Learning rate:{}'.format(learn_rate))
    logger.info('Decay rate of learning rate:{}'.format(decay_rate))
    logger.info('Number of hidden layers:{}'.format(n_hidden_layers))
    logger.info('Number of hidden units:{}'.format(hidden_units))
    logger.info('Dropout rates:{}'.format(dropout_rates))
    logger.info('Early stoping:{}'.format(early_stop))
    logger.info('Retrain model:{}'.format(retrain))
    logger.info('Warm up:{}'.format(warm_up))
    logger.info('Initial epoch of warm up:{}'.format(initial_epoch))
    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    # Set project parameters

    predictor = 'lstm'
    predict_pattern = predict_pattern # hindcast or forecast
    signals = station+'_'+decomposer
    # Set the model id
    model_id=model_id
    if decomposer=='dwt' or decomposer=='modwt':
        data_path = root_path + '/'+signals+'/data/'+wavelet_level+'/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+wavelet_level+'/'+predict_pattern+'/s'+str(model_id)+'/history/'
    else:
        data_path = root_path + '/'+signals+'/data/'+predict_pattern+'/'
        model_path = root_path+'/'+signals+'/projects/'+predictor+'/'+predict_pattern+'/s'+str(model_id)+'/history/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info('Data path:{}'.format(data_path))
    logger.info('Model path:{}'.format(model_path))
    ######################################################
    
    logger.info('Load learning samples...')
    # 1.Import the sampled normalized data set from disk
    train = pd.read_csv(data_path+'minmax_unsample_train_s'+str(model_id)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_s'+str(model_id)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_s'+str(model_id)+'.csv')
    # Split features from labels
    train_x = train
    train_y = train.pop('Y')
    train_y = train_y.as_matrix()
    dev_x = dev
    dev_y = dev.pop('Y')
    dev_y = dev_y.as_matrix()
    test_x = test
    test_y = test.pop('Y')
    test_y = test_y.as_matrix()
    # reshape the input features for LSTM
    train_x = (train_x.values).reshape(train_x.shape[0],1,train_x.shape[1])
    dev_x = (dev_x.values).reshape(dev_x.shape[0],1,dev_x.shape[1])
    test_x = (test_x.values).reshape(test_x.shape[0],1,test_x.shape[1])
    # 2.Build LSTM model with keras
    
    model_name = 'LSTM-S'+str(model_id)+\
        '-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(n_epochs)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    # RESUME_TRAINING = True
    def build_model():
        logger.info('Build LSTM model...')
        if n_hidden_layers==2:
            model = keras.Sequential(
            [
                layers.LSTM(hidden_units[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                layers.LSTM(hidden_units[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(dropout_rates[1], noise_shape=None, seed=seed),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(hidden_units[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(dropout_rates[0], noise_shape=None, seed=seed),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(learn_rate,decay=decay_rate)
        model.compile(loss='mean_squared_error',optimizer=optimizer,
                        metrics=['mean_absolute_error','mean_squared_error'])
        return model

    logger.info('Set model parameters restore path...')
    # set model's parameters restore path
    cp_path = model_path+model_name+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+model_name+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    logger.info('checkpoint dir:{}'.format(checkpoint_dir))
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,mode='min',save_weights_only=True,verbose=1)
    # cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,period=5,verbose=1)
    # if not RESUME_TRAINING:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(checkpoint_dir, ignore_errors=True)
    # else:
    #     print("Resuming training...")
    # initialize a new model
    model = build_model()
    model.summary() #print a simple description for the model
    """
    # Evaluate before training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    # Try the model with initial weights and biases
    example_batch = train_x[:10]
    example_result = model.predict(example_batch)
    print(example_result)
    """
    # 3.Train the model
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
    files = os.listdir(checkpoint_dir)

    
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',min_lr=0.00001,factor=0.2, verbose=1,patience=10, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=200,restore_best_weights=True)

    warm_dir = 'LSTM-S'+str(model_id)+\
        '-LR['+str(learn_rate)+\
        ']-HU'+str(hidden_units)+\
        '-EPS['+str(initial_epoch)+\
        ']-BS['+str(batch_size)+\
        ']-DR'+str(dropout_rates)+\
        '-DC['+str(decay_rate)+\
        ']-SEED['+str(seed)+']'
    logger.info("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    # Training models
    logger.info('Train the LSTM model...')
    if  retrain: # Retraining the LSTM model
        print('retrain the model')
        if early_stop:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,
            batch_size=batch_size ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
                early_stopping,
            ])
            end = time.process_time()
            time_cost = end -start
        else:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=n_epochs,
            batch_size=batch_size ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
            ])
            end = time.process_time()
            time_cost = end - start
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        logger.debug(hist.tail())
        plot_history(history,
        model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
        model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
    elif len(files)==0: # The current model has not been trained
        # Training the model using the trained weights and biases as initialized parameters
        if os.path.exists(model_path+warm_dir) and warm_up:
            # Warm up from the last epoch of the target model
            print('WARM UP FROM EPOCH '+str(initial_epoch))
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if early_stop:
                start = time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=initial_epoch,epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end -start + prev_time_cost
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=initial_epoch,epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            logger.debug(hist.tail())
            plot_history(history,
            model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
            model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
        else:
            logger.info('new train')
            if early_stop:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end - start
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=n_epochs,
                batch_size=batch_size ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end-start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+model_name+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            logger.debug(hist.tail())
            plot_history(history,
            model_path+model_name+'-MAE-ERRORS-TRAINTEST.png',
            model_path+model_name+'-MSE-ERRORS-TRAINTEST.png')
    else:
        logger.info('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+model_name+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)

        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    # 4. Predict the model
    # load the unsample data
    logger.info('Predict the training, development and testing samples...')
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id_s'+str(model_id)+'.csv')
    sMax = norm['series_max'][norm.shape[0]-1]
    sMin = norm['series_min'][norm.shape[0]-1]
    print('Series min:{}'.format(sMin))
    print('Series max:{}'.format(sMax))

    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1,sMax - sMin) / 2 + sMin
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1,sMax - sMin) / 2 + sMin
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1,sMax - sMin) / 2 + sMin

    logger.info('Dump prediction results...')
    dum_pred_results(
        path = model_path+model_name+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    logger.info('Plot the prediction results...')
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + model_name + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + model_name + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + model_name + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+model_name+'-TEST-ERROR-DSTRI.png')
    plt.close('all')