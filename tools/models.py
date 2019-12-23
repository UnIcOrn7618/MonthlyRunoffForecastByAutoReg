#### import basic external libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.set_cmap("viridis")
import datetime
import time
import os
import sys
#### import libs for optimize SVR or GBRT
from sklearn.svm import SVR,NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from functools import partial
from skopt import gp_minimize,forest_minimize, dummy_minimize
from skopt.plots import plot_convergence,plot_objective,plot_evaluations
from sklearn.externals import joblib
from skopt import dump, load
# from skopt.callbacks import CheckpointSaver

#### import libs for building tensorflow
# set parameters to make the program not use GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping


# import own coding libs
from plot_utils import plot_convergence_
from plot_utils import plot_evaluations_
from plot_utils import plot_objective_
from plot_utils import plot_rela_pred
from plot_utils import plot_history
from plot_utils import plot_error_distribution
from dump_data import dum_pred_results

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


def multi_optimizer_esvr(root_path,station,n_calls=100):
    """
    """
    # Set the time series and model parameters
    STATION = station
    PREDICTOR = 'esvr'
    data_path = root_path + '/'+STATION+'/data/'
    model_path = root_path+'/'+STATION+'/projects/'+PREDICTOR+'/multi_optimizer/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+PREDICTOR
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

    
    if os.path.exists(model_path +MODEL_NAME+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path +MODEL_NAME+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            sys.exit()


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



    reg = SVR(tol=1e-4)

    # Set the space of hyper-parameters for tuning them
    space = ESVR_SPACE

    # Define an objective function of hyper-parameters tuning
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))
    
    def run(minimizer, n_iter=5):
        return [minimizer(objective, space, n_calls=n_calls, random_state=n) 
                for n in range(n_iter)]
    #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)
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
    plt.show()

def esvr(root_path,station,optimizer='gp_minimize',n_calls=100):
    """
    """
    # Set the time series and model parameters
    STATION = station
    PREDICTOR = 'esvr'
    data_path = root_path + '/'+STATION+'/data/'
    model_path = root_path+'/'+STATION+'/projects/'+PREDICTOR+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+PREDICTOR
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

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

    if os.path.exists(model_path +MODEL_NAME+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path +MODEL_NAME+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            
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
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))
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
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
                )
            sys.exit()


    reg = SVR(tol=1e-4)
    # Set the space of hyper-parameters for tuning them
    space = ESVR_SPACE

    # Define an objective function of hyper-parameters tuning
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))
    # Tuning the hyper-parameters using Bayesian Optimization based on Gaussion Process
    start = time.process_time()
    if optimizer=='gp_minimize':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
    elif optimizer=='forest_minimize_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
    elif optimizer=='forecast_minimize_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
    elif optimizer=='dummy_minimize':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end = time.process_time()
    time_cost = end-start
    dump(res,model_path+'result.pkl',store_objective=False)
    returned_results = load(model_path+'result.pkl')

    # Visualizing the results of hyper-parameaters tuning
    plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')

    # Plot the optimal hyperparameters
    print('Best score=%.4f'%res.fun)
    print(""" Best parameters:
     -C = %.8f
     -epsilon = %.8f
     -gamma = %.8f
     """%(res.x[0],res.x[1],res.x[2]))
    print('Time cost:{} seconds'.format(time_cost))

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
    params_df.to_csv(model_path +MODEL_NAME+'_optimized_params.csv')
    
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
    print('Series Min:\n {}'.format(sMin))
    print('Series Max:\n {}'.format(sMax))

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
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost = time_cost,
        )

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+MODEL_NAME+"_test_error.png")
    plt.show()
    # Save figures in EPS for manuscript
    plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
    plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+MODEL_NAME+"_test_error.eps",format='EPS',dpi=EPS_DPI)

def esvr_multi_seed(root_path,station,optimizer='gp_minimize',n_calls=100,iterations=10):
    """
    """
    # Set the time series and model parameters
    STATION = station
    PREDICTOR = 'esvr'
    data_path = root_path + '/'+STATION+'/data/'
    model_path = root_path+'/'+STATION+'/projects/'+PREDICTOR+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))
        
    for random_state in range(1,iterations+1):
        MODEL_NAME = STATION+'_'+PREDICTOR+'_seed'+str(random_state)
        print('Model Name:{}'.format(MODEL_NAME))
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
        print("Optimized params:{}".format(model_path +MODEL_NAME+'_optimized_params.csv'))

        if os.path.exists(model_path +MODEL_NAME+'_optimized_params.csv'):
            optimal_params = pd.read_csv(model_path +MODEL_NAME+'_optimized_params.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                print("The n_calls="+str(n_calls)+" was already tuned")
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
                print('Series Min:\n {}'.format(sMin))
                print('Series Max:\n {}'.format(sMax))
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
                    path = model_path+MODEL_NAME+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                    )
                # sys.exit()

        else:
            reg = SVR(tol=1e-4)
            # Set the space of hyper-parameters for tuning them
            space = ESVR_SPACE
            # Define an objective function of hyper-parameters tuning
            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))
            # Tuning the hyper-parameters using Bayesian Optimization based on Gaussion Process
            start = time.process_time()
            if optimizer=='gp_minimize':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=random_state,verbose=True)
            elif optimizer=='forest_minimize_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=random_state,verbose=True)
            elif optimizer=='forecast_minimize_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=random_state,verbose=True)
            elif optimizer=='dummy_minimize':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end = time.process_time()
            time_cost = end-start
            dump(res,model_path+'result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+'result_seed'+str(random_state)+'.pkl')
            # Visualizing the results of hyper-parameaters tuning
            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
            # Plot the optimal hyperparameters
            print('Best score=%.4f'%res.fun)
            print(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))
            print('Time cost:{} seconds'.format(time_cost))
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
            params_df.to_csv(model_path +MODEL_NAME+'_optimized_params.csv')
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
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))
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
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = time_cost,
                )
            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path+MODEL_NAME+"_test_error.png")
        

def one_step_esvr(root_path,station,decomposer,predict_pattern,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100):
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'esvr'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
    SIGNALS = STATION+'_'+DECOMPOSER
    if DECOMPOSER == 'wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

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
    if os.path.exists(model_path + MODEL_NAME+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path + MODEL_NAME+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
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
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))
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
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
            )
            sys.exit()


    reg = SVR(tol=1e-4)

    space = ESVR_SPACE

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

    #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

    start = time.process_time()
    if optimizer=='gp_minimize':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
    elif optimizer=='forest_minimize_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
    elif optimizer=='forecast_minimize_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
    elif optimizer=='dummy_minimize':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end = time.process_time()
    time_cost = end -start
    dump(res,model_path+'result.pkl',store_objective=False)
    returned_results = load(model_path+'result.pkl')
    
    plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
    
    print('Best score=%.4f'%res.fun)
    print(""" Best parameters:
     -C = %.8f
     -epsilon = %.8f
     -gamma = %.8f
     """%(res.x[0],res.x[1],res.x[2]))

    print('Time cost:{}'.format(time_cost))
    params_dict={
        'C':res.x[0],
        'epsilon':res.x[1],
        'gamma':res.x[2],
        'time_cost':(time_cost),
        'n_calls':n_calls,
    }

    params_df = pd.DataFrame(params_dict,index=[0])
    params_df.to_csv(model_path + MODEL_NAME+'_optimized_params.csv')

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
    print('Series Min:\n {}'.format(sMin))
    print('Series Max:\n {}'.format(sMax))

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
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_error.png")
    plt.show()
    
    plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
    plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)

def one_step_esvr_multi_seed(root_path,station,decomposer,predict_pattern,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100,iterations=10):
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'esvr'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast or hindcast_with_pca_mle or forecast_with_pca_mle
    SIGNALS = STATION+'_'+DECOMPOSER
    if DECOMPOSER == 'wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

    for random_state in range(1,iterations+1):
        MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN+'_seed'+str(random_state)
        print('Model Name:{}'.format(MODEL_NAME))
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
        print("Optimized params:{}".format(model_path +MODEL_NAME+'_optimized_params.csv'))
        if os.path.exists(model_path + MODEL_NAME+'_optimized_params.csv'):
            optimal_params = pd.read_csv(model_path + MODEL_NAME+'_optimized_params.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                print("The n_calls="+str(n_calls)+" was already tuned")
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
                print('Series Min:\n {}'.format(sMin))
                print('Series Max:\n {}'.format(sMax))
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
                    path = model_path+MODEL_NAME+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                )
                # sys.exit()
        else:
            reg = SVR(tol=1e-4)

            space = ESVR_SPACE

            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

            #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

            start = time.process_time()
            if optimizer=='gp_minimize':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=random_state,verbose=True)
            elif optimizer=='forest_minimize_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=random_state,verbose=True)
            elif optimizer=='forecast_minimize_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=random_state,verbose=True)
            elif optimizer=='dummy_minimize':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end = time.process_time()
            time_cost = end -start
            dump(res,model_path+'result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+'result_seed'+str(random_state)+'.pkl')

            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')

            print('Best score=%.4f'%res.fun)
            print(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))

            print('Time cost:{}'.format(time_cost))
            params_dict={
                'C':res.x[0],
                'epsilon':res.x[1],
                'gamma':res.x[2],
                'time_cost':(time_cost),
                'n_calls':n_calls,
            }

            params_df = pd.DataFrame(params_dict,index=[0])
            params_df.to_csv(model_path + MODEL_NAME+'_optimized_params.csv')

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
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))

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
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost=time_cost)
            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_error.png")
            # plt.show()

        

def multi_step_esvr(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100):

    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'esvr'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    MODEL_ID = model_id
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN+'_imf'+str(MODEL_ID)
    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))
    train = pd.read_csv(data_path+'minmax_unsample_train_imf'+str(MODEL_ID)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_imf'+str(MODEL_ID)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_imf'+str(MODEL_ID)+'.csv')
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
    print("Optimized params:{}".format(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv'))
    if os.path.exists(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv'):
        optimal_params = pd.read_csv(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()
            norm_id = pd.read_csv(data_path + 'norm_unsample_id_imf'+str(MODEL_ID)+'.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))
            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
            dum_pred_results(
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost = optimal_params['time_cost'][0],
            )
            # sys.exit()
    else:
        reg = SVR(tol=1e-4)
    
        space = ESVR_SPACE
    
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))
    
        #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)
    
        start = time.process_time()
        if optimizer=='gp_minimize':
            res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
        elif optimizer=='forest_minimize_bt':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
        elif optimizer=='forecast_minimize_rf':
            res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
        elif optimizer=='dummy_minimize':
            res = dummy_minimize(objective,space,n_calls=n_calls)
        end=time.process_time()
        time_cost = end -start
        dump(res,model_path+'result.pkl',store_objective=False)
        returned_results = load(model_path+'result.pkl')
    
        plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
        plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
        plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
        
        
        print('Best score=%.4f'%res.fun)
        print(""" Best parameters:
         -C = %.8f
         -epsilon = %.8f
         -gamma = %.8f
         """%(res.x[0],res.x[1],res.x[2]))
    
        print('Time cost:{}'.format(time_cost))
        params_dict={
            'C':res.x[0],
            'epsilon':res.x[1],
            'gamma':res.x[2],
            'time_cost':(time_cost),
            'n_calls':n_calls,
        }
    
        params_df = pd.DataFrame(params_dict,index=[0])
        params_df.to_csv(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv')
    
        esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
        # Do prediction with the opyimal model
        train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
        dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
        test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
    
        train_y=(train_y.values).flatten()
        dev_y=(dev_y.values).flatten()
        test_y=(test_y.values).flatten()
    
        norm_id = pd.read_csv(data_path + 'norm_unsample_id_imf' + str(MODEL_ID) + '.csv')
        sMin = norm_id['series_min'][norm_id.shape[0]-1]
        sMax = norm_id['series_max'][norm_id.shape[0]-1]
        print('Series Min:\n {}'.format(sMin))
        print('Series Max:\n {}'.format(sMax))
    
        # Renormalized the records and predictions
        train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
        dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
        test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
        train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
        dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
        test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
    
    
        dum_pred_results(
            path = model_path+MODEL_NAME+'.csv',
            train_y = train_y,
            train_predictions=train_predictions,
            dev_y = dev_y,
            dev_predictions = dev_predictions,
            test_y = test_y,
            test_predictions = test_predictions,
            time_cost=time_cost)
    
        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.png')
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.png")
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.png")
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.png",)
        plt.show()
        plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
        plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
        plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
        plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
        plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
        plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
        plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)


def multi_step_esvr_multi_seed(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100,iterations=10):

    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'esvr'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    MODEL_ID = model_id
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for random_state in range(1,iterations+1):
        MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN+'_imf'+str(MODEL_ID)+'_seed'+str(random_state)
        print("Data Path:{}".format(data_path))
        print("Model Path:{}".format(model_path))

        train = pd.read_csv(data_path+'minmax_unsample_train_imf'+str(MODEL_ID)+'.csv')
        dev = pd.read_csv(data_path+'minmax_unsample_dev_imf'+str(MODEL_ID)+'.csv')
        test = pd.read_csv(data_path+'minmax_unsample_test_imf'+str(MODEL_ID)+'.csv')
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
        print("Optimized params:{}".format(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv'))
        if os.path.exists(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv'):
            optimal_params = pd.read_csv(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv')
            pre_n_calls = optimal_params['n_calls'][0]
            if pre_n_calls==n_calls:
                print("The n_calls="+str(n_calls)+" was already tuned")
                esvr = SVR(C=optimal_params['C'][0], epsilon=optimal_params['epsilon'][0], gamma=optimal_params['gamma'][0])
                # Do prediction with the opyimal model
                train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
                dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
                test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)
                train_y=(train_y.values).flatten()
                dev_y=(dev_y.values).flatten()
                test_y=(test_y.values).flatten()
                norm_id = pd.read_csv(data_path + 'norm_unsample_id_imf'+str(MODEL_ID)+'.csv')
                sMin = norm_id['series_min'][norm_id.shape[0]-1]
                sMax = norm_id['series_max'][norm_id.shape[0]-1]
                print('Series Min:\n {}'.format(sMin))
                print('Series Max:\n {}'.format(sMax))
                # Renormalized the records and predictions
                train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
                dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
                test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
                train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
                dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
                test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin
                dum_pred_results(
                    path = model_path+MODEL_NAME+'.csv',
                    train_y = train_y,
                    train_predictions=train_predictions,
                    dev_y = dev_y,
                    dev_predictions = dev_predictions,
                    test_y = test_y,
                    test_predictions = test_predictions,
                    time_cost = optimal_params['time_cost'][0],
                )
                # sys.exit()
        else:
            reg = SVR(tol=1e-4)

            space = ESVR_SPACE

            @use_named_args(space)
            def objective(**params):
                reg.set_params(**params)
                return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

            #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

            start = time.process_time()
            if optimizer=='gp_minimize':
                res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
            elif optimizer=='forest_minimize_bt':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
            elif optimizer=='forecast_minimize_rf':
                res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
            elif optimizer=='dummy_minimize':
                res = dummy_minimize(objective,space,n_calls=n_calls)
            end=time.process_time()
            time_cost = end -start
            dump(res,model_path+'result_seed'+str(random_state)+'.pkl',store_objective=False)
            returned_results = load(model_path+'result_seed'+str(random_state)+'.pkl')
            
            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.png')
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
            plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')


            print('Best score=%.4f'%res.fun)
            print(""" Best parameters:
             -C = %.8f
             -epsilon = %.8f
             -gamma = %.8f
             """%(res.x[0],res.x[1],res.x[2]))

            print('Time cost:{}'.format(time_cost))
            params_dict={
                'C':res.x[0],
                'epsilon':res.x[1],
                'gamma':res.x[2],
                'time_cost':(time_cost),
                'n_calls':n_calls,
            }

            params_df = pd.DataFrame(params_dict,index=[0])
            params_df.to_csv(model_path + MODEL_NAME +'_optimized_params_imf' + str(MODEL_ID) +'.csv')

            esvr = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2])
            # Do prediction with the opyimal model
            train_predictions = esvr.fit(train_dev_x,train_dev_y).predict(train_x)
            dev_predictions = esvr.fit(train_dev_x,train_dev_y).predict(dev_x)
            test_predictions = esvr.fit(train_dev_x,train_dev_y).predict(test_x)

            train_y=(train_y.values).flatten()
            dev_y=(dev_y.values).flatten()
            test_y=(test_y.values).flatten()

            norm_id = pd.read_csv(data_path + 'norm_unsample_id_imf' + str(MODEL_ID) + '.csv')
            sMin = norm_id['series_min'][norm_id.shape[0]-1]
            sMax = norm_id['series_max'][norm_id.shape[0]-1]
            print('Series Min:\n {}'.format(sMin))
            print('Series Max:\n {}'.format(sMax))

            # Renormalized the records and predictions
            train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
            dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
            test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
            train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
            dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
            test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin


            dum_pred_results(
                path = model_path+MODEL_NAME+'.csv',
                train_y = train_y,
                train_predictions=train_predictions,
                dev_y = dev_y,
                dev_predictions = dev_predictions,
                test_y = test_y,
                test_predictions = test_predictions,
                time_cost=time_cost)

            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.png')
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.png")
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.png")
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.png",)
            plt.show()
            plot_objective_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
            plot_evaluations_(res,dimensions=DIMENSION_ESVR,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
            plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
            plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
            plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
            plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
            plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)


def gbrt(root_path,station,optimizer='gp_minimize',n_calls=100):
    """
    """
    STATION = station
    PREDICTOR = 'gbrt'
    data_path = root_path + '/'+STATION+'/data/'
    model_path = root_path+'/'+STATION+'/projects/'+PREDICTOR+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+PREDICTOR

    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

    
    if os.path.exists(model_path +MODEL_NAME+'_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path +MODEL_NAME+'_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            sys.exit()

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
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

    #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

    start = time.process_time()
    if optimizer=='gp_minimize':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
    elif optimizer=='forest_minimize_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
    elif optimizer=='forecast_minimize_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
    elif optimizer=='dummy_minimize':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end=time.process_time()
    time_cost = end-start
    dump(res,model_path+'result.pkl',store_objective=False)
    returned_results = load(model_path+'result.pkl')
    
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
    

    print('Best score=%.4f'%res.fun)
    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                res.x[4]))
    # end=datetime.datetime.now()
    print('Time cost:{}'.format(time_cost))

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
    params_df.to_csv(model_path +MODEL_NAME+'_optimized_params.csv')

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
    print('Series Min:\n {}'.format(sMin))
    print('Series Max:\n {}'.format(sMax))

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
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_error.png")
    plt.show()
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path +MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path +MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path +MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)

def one_step_gbrt(root_path,station,decomposer,predict_pattern,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100):

    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'gbrt'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN

    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

    
    if os.path.exists(model_path + MODEL_NAME+ '_optimized_params.csv'):
        optimal_params = pd.read_csv(model_path + MODEL_NAME+ '_optimized_params.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            sys.exit()

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
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

    #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

    start = time.process_time()
    if optimizer=='gp_minimize':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
    elif optimizer=='forest_minimize_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
    elif optimizer=='forecast_minimize_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
    elif optimizer=='dummy_minimize':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end=time.process_time()
    time_cost = end - start
    dump(res,model_path+'result.pkl',store_objective=False)
    returned_results = load(model_path+'result.pkl')

    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
    
    
    print('Best score=%.4f'%res.fun)
    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                res.x[4]))
    # end=datetime.datetime.now()
    print('Time cost:{}'.format(time_cost))

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
    params_df.to_csv(model_path + MODEL_NAME+ '_optimized_params.csv')

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
    print('Series Min:\n {}'.format(sMin))
    print('Series Max:\n {}'.format(sMax))

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
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.png",)
    plt.show()
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)


def multi_step_gbrt(root_path,station,decomposer,predict_pattern,lags,model_id,optimizer='gp_minimize',wavelet_level='db10-lev2',n_calls=100):

    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'gbrt'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    # Set the mode id:
    MODEL_ID = model_id
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    MODEL_NAME = STATION+'_'+DECOMPOSER+'_'+PREDICTOR+'_'+PREDICT_PATTERN+'_imf'+str(MODEL_ID)

    print("Data Path:{}".format(data_path))
    print("Model Path:{}".format(model_path))

    
    if os.path.exists(model_path + MODEL_NAME+'_optimized_params_imf' + str(MODEL_ID) +'.csv'):
        optimal_params = pd.read_csv(model_path + MODEL_NAME+'_optimized_params_imf' + str(MODEL_ID) +'.csv')
        pre_n_calls = optimal_params['n_calls'][0]
        if pre_n_calls==n_calls:
            print("The n_calls="+str(n_calls)+" was already tuned")
            sys.exit()

    # load data
    train = pd.read_csv(data_path+'minmax_unsample_train_imf'+str(MODEL_ID)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_imf'+str(MODEL_ID)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_imf'+str(MODEL_ID)+'.csv')
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
        return -np.mean(cross_val_score(reg,train_dev_x,train_dev_y,cv=6,n_jobs=-1,scoring='neg_mean_squared_error'))

    #checkpoint_saver = CheckpointSaver(model_path+MODEL_NAME+'/checkpoint.pkl',compress=9)

    start = time.process_time()
    if optimizer=='gp_minimize':
        res = gp_minimize(objective,space,n_calls=n_calls ,random_state=0,verbose=True)
    elif optimizer=='forest_minimize_bt':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='ET',random_state=0,verbose=True)
    elif optimizer=='forecast_minimize_rf':
        res = forest_minimize(objective,space,n_calls=n_calls,base_estimator='RF',random_state=0,verbose=True)
    elif optimizer=='dummy_minimize':
        res = dummy_minimize(objective,space,n_calls=n_calls)
    end=time.process_time()
    time_cost = end -start
    dump(res,model_path+'result.pkl',store_objective=False)
    returned_results = load(model_path+'result.pkl')
    
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.png')
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.png')
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.png')
    

    print('Best score=%.4f'%res.fun)
    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res.x[0], res.x[1], res.x[2], res.x[3],
                                res.x[4]))
    # end=datetime.datetime.now()
    print('Time cost:{}'.format(time_cost))

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
    params_df.to_csv(model_path + MODEL_NAME+'_optimized_params_imf' + str(MODEL_ID) +'.csv')

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

    norm_id = pd.read_csv(data_path + 'norm_unsample_id_imf' + str(MODEL_ID) + '.csv')
    sMin = norm_id['series_min'][norm_id.shape[0]-1]
    sMax = norm_id['series_max'][norm_id.shape[0]-1]
    print('Series Min:\n {}'.format(sMin))
    print('Series Max:\n {}'.format(sMax))

    # Renormalized the records and predictions
    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1, sMax -sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1, sMax -sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1, sMax -sMin) / 2 + sMin


    dum_pred_results(
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.png")
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.png")
    plt.show()
    plot_objective_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_objective.eps',format='EPS',dpi=EPS_DPI)
    plot_evaluations_(res,dimensions=DIMENSION_GBRT,fig_savepath=model_path+MODEL_NAME+'_evaluation.eps',format='EPS',dpi=EPS_DPI)
    plot_convergence_(res,fig_savepath=model_path+MODEL_NAME+'_convergence.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '_train_pred.eps',format='EPS',dpi=EPS_DPI)
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "_dev_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_pred.eps",format='EPS',dpi=EPS_DPI)
    plot_error_distribution(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "_test_error.eps",format='EPS',dpi=EPS_DPI)



def lstm(root_path,station,seed,
    epochs_num=5000,
    batch_size=128,
    learning_rate=0.007,
    decay_rate=0.0,
    hidden_layer=1,
    hidden_units_1=8,
    dropout_rate_1=0.0,
    hidden_units_2=8,
    dropout_rate_2=0.0,
    early_stoping=True,
    retrain=False,
    warm_up=False,
    initial_epoch=None,
    ):

    STATION = station
    PREDICTOR = 'lstm'
    data_path = root_path + '/'+STATION+'/data/'
    model_path = root_path+'/'+STATION+'/projects/'+PREDICTOR+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    RE_TRAIN = retrain
    WARM_UP = warm_up
    EARLY_STOPING = early_stoping
    INITIAL_EPOCH = initial_epoch

    # For initialize weights and bias
    SEED=seed
    # set hyper-parameters
    EPS=epochs_num    #epochs number 500 for learning rate analysis
    #########--1--###########
    LR=learning_rate    #learnin rate 0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007,0.01, 0.03 ,0.07,0.1
    #########--2--############
    HU1 = hidden_units_1    #hidden units for hidden layer 1
    BS = batch_size   #batch size
    #########--3--###########
    HL = hidden_layer      #hidden layers
    HU2 = hidden_units_2    #hidden units for hidden layer 2
    DC=decay_rate    #decay rate of learning rate
    #########--4--###########
    DR1=dropout_rate_1    #dropout rate for hidden layer 1
    DR2=dropout_rate_1     #dropout rate for hidden layer 2

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
    # set the hyper-parameters
    LEARNING_RATE=LR
    EPOCHS = EPS
    BATCH_SIZE = BS
    if HL==2:
        HIDDEN_UNITS = [HU1,HU2]
        DROP_RATE = [DR1,DR2]
    else:
        HIDDEN_UNITS = [HU1]
        DROP_RATE = [DR1]
    DECAY_RATE = DC
    MODEL_NAME = 'LSTM-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(EPOCHS)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    # RESUME_TRAINING = True
    def build_model():
        if HL==2:
            model = keras.Sequential(
            [
                layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                layers.LSTM(HIDDEN_UNITS[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(DROP_RATE[1], noise_shape=None, seed=None),
                # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                    # layers.LSTM(HIDDEN_UNITS1,activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])), # first hidden layer if hasnext hidden layer
                    # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(LEARNING_RATE,
        decay=DECAY_RATE
        )
        model.compile(loss='mean_squared_error',
                        optimizer=optimizer,
                        metrics=['mean_absolute_error','mean_squared_error'])
        return model
    # set model's parameters restore path
    cp_path = model_path+MODEL_NAME+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+MODEL_NAME+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print('checkpoint dir:{}'.format(checkpoint_dir))
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


    warm_dir = 'LSTM-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(INITIAL_EPOCH)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    print("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    # Training models
    if  RE_TRAIN: # Retraining the LSTM model
        print('retrain the model')
        if EARLY_STOPING:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
            callbacks=[
                cp_callback,
                early_stopping,
            ])
            end = time.process_time()
            time_cost = end-start
        else:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[cp_callback])
            end =time.process_time()
            time_cost = end-start
        # # Visualize the model's training progress using the stats stored in the history object
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    elif len(files)==0: # The current model has not been trained
        if os.path.exists(model_path+warm_dir) and WARM_UP: # Training the model using the trained weights and biases as initialized parameters
            print('WARM UP FROM EPOCH '+str(INITIAL_EPOCH)) # Warm up from the last epoch of the target model
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if EARLY_STOPING:
                start=time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
        else: # Training entirely new model
            print('new train')
            if EARLY_STOPING:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,callbacks=[
                    cp_callback,
                    early_stopping,
                    ])
                end = time.process_time()
                time_cost = end -start
            else:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=EPOCHS,batch_size=BATCH_SIZE ,validation_data=(dev_x,dev_y),verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    else:
        print('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+MODEL_NAME+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)
        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    # 4. Predict the model
    # load the unsample data
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id.csv')
    sMax = norm['series_max'][norm.shape[0]-1]
    sMin = norm['series_min'][norm.shape[0]-1]
    print('Series min:{}'.format(sMin))
    print('Series max:{}'.format(sMax))

    train_y = np.multiply(train_y + 1,sMax - sMin) / 2 + sMin
    train_predictions = np.multiply(train_predictions + 1,sMax - sMin) / 2 + sMin
    train_predictions[train_predictions<0.0]=0.0
    dev_y = np.multiply(dev_y + 1,sMax - sMin) / 2 + sMin
    dev_predictions = np.multiply(dev_predictions + 1,sMax - sMin) / 2 + sMin
    dev_predictions[dev_predictions<0.0]=0.0
    test_y = np.multiply(test_y + 1,sMax - sMin) / 2 + sMin
    test_predictions = np.multiply(test_predictions + 1,sMax - sMin) / 2 + sMin
    test_predictions[test_predictions<0.0]=0.0

    dum_pred_results(
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost,
        )

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+MODEL_NAME+'-TEST-ERROR-DSTRI.png')


def one_step_lstm(
        root_path,station,decomposer,predict_pattern,seed,
        wavelet_level='db10-lev2',
        epochs_num=5000,
        batch_size=128,
        learning_rate=0.007,
        decay_rate=0.0,
        hidden_layer=1,
        hidden_units_1=8,
        dropout_rate_1=0.0,
        hidden_units_2=8,
        dropout_rate_2=0.0,
        early_stoping=True,
        retrain=False,
        warm_up=False,
        initial_epoch=None,
    ):
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'lstm'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/one_step_one_month_'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/one_step_one_month_'+PREDICT_PATTERN+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/one_step_one_month_'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/one_step_one_month_'+PREDICT_PATTERN+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ######################################################
    RE_TRAIN = retrain
    WARM_UP = warm_up
    EARLY_STOPING = early_stoping
    INITIAL_EPOCH = initial_epoch
    # For initialize weights and bias
    SEED=seed
    # set hyper-parameters
    EPS=epochs_num     #epochs number
    #########--1--###########
    LR=learning_rate     #learnin rate 0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007,0.01, 0.03 0.1
    #########--2--############
    HU1 = hidden_units_1     #hidden units for hidden layer 1
    BS = batch_size    #batch size
    #########--3--###########
    HL = hidden_layer      #hidden layers
    HU2 = hidden_units_2    #hidden units for hidden layer 2
    DC=decay_rate  #decay rate of learning rate
    #########--4--###########
    DR1=dropout_rate_1      #dropout rate for hidden layer 1
    DR2=dropout_rate_2      #dropout rate for hidden layer 2
    ########################################################

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
    # set the hyper-parameters
    LEARNING_RATE=LR
    EPOCHS = EPS
    BATCH_SIZE = BS
    if HL==2:
        HIDDEN_UNITS = [HU1,HU2]
        DROP_RATE = [DR1,DR2]
    else:
        HIDDEN_UNITS = [HU1]
        DROP_RATE = [DR1]

    DECAY_RATE = DC
    MODEL_NAME = 'LSTM-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(EPOCHS)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    # RESUME_TRAINING = True
    def build_model():
        if HL==2:
            model = keras.Sequential(
            [
                layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                layers.LSTM(HIDDEN_UNITS[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(DROP_RATE[1], noise_shape=None, seed=None),
                # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                    # layers.LSTM(HIDDEN_UNITS1,activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])), # first hidden layer if hasnext hidden layer
                    # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(LEARNING_RATE,
        decay=DECAY_RATE
        )
        model.compile(loss='mean_squared_error',
                        optimizer=optimizer,
                        metrics=['mean_absolute_error','mean_squared_error'])
        return model
    # set model's parameters restore path
    cp_path = model_path+MODEL_NAME+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+MODEL_NAME+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print('checkpoint dir:{}'.format(checkpoint_dir))
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


    warm_dir = 'LSTM-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(INITIAL_EPOCH)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    print("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    # Training models
    if  RE_TRAIN: # Retraining the LSTM model
        print('retrain the model')
        if EARLY_STOPING:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,
            batch_size=BATCH_SIZE ,
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
            history = model.fit(train_x,train_y,epochs=EPOCHS,
            batch_size=BATCH_SIZE ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
            ])
            end = time.process_time()
            time_cost = end - start
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,
        model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
        model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')

    elif len(files)==0:# The current model has not been trained
        # Training the model using the trained weights and biases as initialized parameters
        if os.path.exists(model_path+warm_dir) and WARM_UP:
            # Warm up from the last epoch of the target model
            print('WARM UP FROM EPOCH '+str(INITIAL_EPOCH))
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if EARLY_STOPING:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                initial_epoch=INITIAL_EPOCH,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
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
                initial_epoch=INITIAL_EPOCH,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
            model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
        else:
            print('new train')
            if EARLY_STOPING:
                start = time.process_time()
                history = model.fit(train_x,train_y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
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
                epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[cp_callback,])
                end = time.process_time()
                time_cost = end - start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
            model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    else:
        print('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+MODEL_NAME+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)

        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    # 4. Predict the model
    # load the unsample data
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id.csv')
    sMax = norm['series_max'][norm.shape[0]-1]
    sMin = norm['series_min'][norm.shape[0]-1]
    print('Series min:{}'.format(sMin))
    print('Series max:{}'.format(sMax))

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
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost = time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+MODEL_NAME+'-TEST-ERROR-DSTRI.png')



def multi_step_lstm(
    root_path,station,decomposer,predict_pattern,lags,model_id,seed,
    wavelet_level='db10-lev2',
    epochs_num=5000,
    batch_size=128,
    learning_rate=0.007,
    decay_rate=0.0,
    hidden_layer=1,
    hidden_units_1=8,
    dropout_rate_1=0.0,
    hidden_units_2=8,
    dropout_rate_2=0.0,
    early_stoping=True,
    retrain=False,
    warm_up=False,
    initial_epoch=None,
):
    if model_id>len(lags):
        raise Exception("The model id exceed the number of sub-signals")
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICTOR = 'lstm'
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER

    # Set the model id
    MODEL_ID=model_id
    if DECOMPOSER=='wd':
        data_path = root_path + '/'+SIGNALS+'/data/'+wavelet_level+'/multi_step_one_month_'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/'+wavelet_level+'/multi_step_one_month_'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'
    else:
        data_path = root_path + '/'+SIGNALS+'/data/multi_step_one_month_'+PREDICT_PATTERN+'/'
        model_path = root_path+'/'+SIGNALS+'/projects/'+PREDICTOR+'/multi_step_one_month_'+PREDICT_PATTERN+'/imf'+str(MODEL_ID)+'/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ######################################################
    RE_TRAIN = retrain
    WARM_UP = warm_up
    EARLY_STOPING = early_stoping
    INITIAL_EPOCH = initial_epoch
    # For initialize weights and bias
    SEED=seed
    # set hyper-parameters
    EPS=epochs_num     #epochs number
    #########--1--###########
    LR=learning_rate     #learnin rate 0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007,0.01, 0.03 0.1
    #########--2--############
    HU1 = hidden_units_1     #hidden units for hidden layer 1
    BS = batch_size    #batch size
    #########--3--###########
    HL = hidden_layer      #hidden layers
    HU2 = hidden_units_2    #hidden units for hidden layer 2
    DC=decay_rate  #decay rate of learning rate
    #########--4--###########
    DR1=dropout_rate_1      #dropout rate for hidden layer 1
    DR2=dropout_rate_2      #dropout rate for hidden layer 2
    ########################################################

    # 1.Import the sampled normalized data set from disk
    train = pd.read_csv(data_path+'minmax_unsample_train_imf'+str(MODEL_ID)+'.csv')
    dev = pd.read_csv(data_path+'minmax_unsample_dev_imf'+str(MODEL_ID)+'.csv')
    test = pd.read_csv(data_path+'minmax_unsample_test_imf'+str(MODEL_ID)+'.csv')

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
    # set the hyper-parameters
    LEARNING_RATE=LR
    EPOCHS = EPS
    BATCH_SIZE = BS
    if HL==2:
        HIDDEN_UNITS = [HU1,HU2]
        DROP_RATE = [DR1,DR2]
    else:
        HIDDEN_UNITS = [HU1]
        DROP_RATE = [DR1]

    DECAY_RATE = DC
    MODEL_NAME = 'LSTM-IMF'+str(MODEL_ID)+\
        '-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(EPOCHS)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    # RESUME_TRAINING = True
    def build_model():
        if HL==2:
            model = keras.Sequential(
            [
                layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])),
                layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                layers.LSTM(HIDDEN_UNITS[1],activation=tf.nn.relu,return_sequences=False), # first hidden layer if hasnext hidden layer
                layers.Dropout(DROP_RATE[1], noise_shape=None, seed=None),
                # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                layers.Dense(1)
            ]
        )
        else:
            model = keras.Sequential(
                [
                    layers.LSTM(HIDDEN_UNITS[0],activation=tf.nn.relu,input_shape=(train_x.shape[1],train_x.shape[2])),
                    layers.Dropout(DROP_RATE[0], noise_shape=None, seed=None),
                    # layers.LSTM(HIDDEN_UNITS1,activation=tf.nn.relu,return_sequences=True,input_shape=(train_x.shape[1],train_x.shape[2])), # first hidden layer if hasnext hidden layer
                    # layers.LSTM(20,activation=tf.nn.relu,return_sequence=True),
                    layers.Dense(1)
                ]
            )
        optimizer = keras.optimizers.Adam(LEARNING_RATE,
        decay=DECAY_RATE
        )
        model.compile(loss='mean_squared_error',
                        optimizer=optimizer,
                        metrics=['mean_absolute_error','mean_squared_error'])
        return model
    # set model's parameters restore path
    cp_path = model_path+MODEL_NAME+'\\'
    if not os.path.exists(cp_path):
        os.makedirs(cp_path)
    checkpoint_path = model_path+MODEL_NAME+'\\cp.ckpt' #restore only the latest checkpoint after every update
    # checkpoint_path = model_path+'cp-{epoch:04d}.ckpt' #restore the checkpoint every period=x epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print('checkpoint dir:{}'.format(checkpoint_dir))
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


    warm_dir = 'LSTM-IMF'+str(MODEL_ID)+\
        '-LR['+str(LEARNING_RATE)+\
        ']-HU'+str(HIDDEN_UNITS)+\
        '-EPS['+str(INITIAL_EPOCH)+\
        ']-BS['+str(BATCH_SIZE)+\
        ']-DR'+str(DROP_RATE)+\
        '-DC['+str(DECAY_RATE)+\
        ']-SEED['+str(SEED)+']'
    print("WARM UP PATH:{}".format(os.path.exists(model_path+warm_dir)))
    # Training models
    if  RE_TRAIN: # Retraining the LSTM model
        print('retrain the model')
        if EARLY_STOPING:
            start = time.process_time()
            history = model.fit(train_x,train_y,epochs=EPOCHS,
            batch_size=BATCH_SIZE ,
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
            history = model.fit(train_x,train_y,epochs=EPOCHS,
            batch_size=BATCH_SIZE ,
            validation_data=(dev_x,dev_y),
            verbose=1,
            callbacks=[
                cp_callback,
            ])
            end = time.process_time()
            time_cost = end - start
        hist = pd.DataFrame(history.history)
        hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
        hist['epoch']=history.epoch
        # print(hist.tail())
        plot_history(history,
        model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
        model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    elif len(files)==0: # The current model has not been trained
        # Training the model using the trained weights and biases as initialized parameters
        if os.path.exists(model_path+warm_dir) and WARM_UP:
            # Warm up from the last epoch of the target model
            print('WARM UP FROM EPOCH '+str(INITIAL_EPOCH))
            prev_time_cost = (pd.read_csv(model_path+warm_dir+'.csv')['time_cost'])[0]
            warm_path=model_path+warm_dir+'\\cp.ckpt'
            model.load_weights(warm_path)
            if EARLY_STOPING:
                start = time.process_time()
                history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
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
                history = model.fit(train_x,train_y,initial_epoch=INITIAL_EPOCH,epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end - start + prev_time_cost
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
            model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
        else:
            print('new train')
            if EARLY_STOPING:
                start = time.process_time()
                history = model.fit(train_x,train_y,epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
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
                history = model.fit(train_x,train_y,epochs=EPOCHS,
                batch_size=BATCH_SIZE ,
                validation_data=(dev_x,dev_y),
                verbose=1,
                callbacks=[
                    cp_callback,
                    ])
                end = time.process_time()
                time_cost = end-start
            hist = pd.DataFrame(history.history)
            hist.to_csv(model_path+MODEL_NAME+'-HISTORY-TRAIN-TEST.csv')
            hist['epoch']=history.epoch
            # print(hist.tail())
            plot_history(history,
            model_path+MODEL_NAME+'-MAE-ERRORS-TRAINTEST.png',
            model_path+MODEL_NAME+'-MSE-ERRORS-TRAINTEST.png')
    else:
        print('#'*10+'Already Trained')
        time_cost = (pd.read_csv(model_path+MODEL_NAME+'.csv')['time_cost'])[0]
        model.load_weights(checkpoint_path)

        # loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    """
    # Evaluate after training or load trained weights and biases
    loss, mae, mse = model.evaluate(test_x, test_y, verbose=1)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
    """
    # 4. Predict the model
    # load the unsample data
    train_predictions = model.predict(train_x).flatten()
    dev_predictions = model.predict(dev_x).flatten()
    test_predictions = model.predict(test_x).flatten()
    # renormized the predictions and labels
    # load the normalized traindev indicators
    norm = pd.read_csv(data_path+'norm_unsample_id_imf'+str(MODEL_ID)+'.csv')
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

    dum_pred_results(
        path = model_path+MODEL_NAME+'.csv',
        train_y = train_y,
        train_predictions=train_predictions,
        dev_y = dev_y,
        dev_predictions = dev_predictions,
        test_y = test_y,
        test_predictions = test_predictions,
        time_cost=time_cost)

    plot_rela_pred(train_y,train_predictions,fig_savepath=model_path + MODEL_NAME + '-TRAIN-PRED.png')
    plot_rela_pred(dev_y,dev_predictions,fig_savepath=model_path + MODEL_NAME + "-DEV-PRED.png")
    plot_rela_pred(test_y,test_predictions,fig_savepath=model_path + MODEL_NAME + "-TEST-PRED.png")
    plot_error_distribution(test_predictions,test_y,model_path+MODEL_NAME+'-TEST-ERROR-DSTRI.png')
