import numpy as np
import pandas as pd
from sklearn import decomposition
import os

def one_step_pca(root_path,station,decomposer,predict_pattern,n_components='mle',wavelet_level='db10-lev2'):
    # Set project parameters
    STATION = station
    DECOMPOSER = decomposer
    PREDICT_PATTERN = predict_pattern # hindcast or forecast
    SIGNALS = STATION+'_'+DECOMPOSER
    # Set parameters for PCA
    n_components = n_components # mle, int, float
    # load one-step one-month forecast or hindcast samples and the normalization indicators
    if decomposer=='wd':
        train = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_train.csv')
        dev = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_dev.csv')
        test = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_test.csv')
        norm_id = pd.read_csv(root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'/norm_unsample_id.csv')
    else:
        train = pd.read_csv(root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_train.csv')
        dev = pd.read_csv(root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_dev.csv')
        test = pd.read_csv(root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'/minmax_unsample_test.csv')
        norm_id = pd.read_csv(root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'/norm_unsample_id.csv')
    sMax = (norm_id['series_max']).values
    sMin = (norm_id['series_min']).values
    # Conncat the training, development and testing samples
    samples = pd.concat([train,dev,test],axis=0)
    samples = samples.reset_index(drop=True)
    # Renormalized the entire samples
    samples = np.multiply(samples + 1,sMax - sMin) / 2 + sMin

    y = samples['Y']
    X = samples.drop('Y',axis=1)
    print("Input features before PAC:\n{}".format(X.tail()))
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    pca_X = pca.transform(X)
    columns=[]
    for i in range(1,pca_X.shape[1]+1):
        columns.append('X'+str(i))
    pca_X = pd.DataFrame(pca_X,columns=columns)
    print("Input features after PAC:\n{}".format(pca_X.tail()))

    pca_samples = pd.concat([pca_X,y],axis=1)
    pca_train = pca_samples.iloc[:train.shape[0]]
    pca_train=pca_train.reset_index(drop=True)
    pca_dev = pca_samples.iloc[train.shape[0]:train.shape[0]+dev.shape[0]]
    pca_dev=pca_dev.reset_index(drop=True)
    pca_test = pca_samples.iloc[train.shape[0]+dev.shape[0]:]
    pca_test=pca_test.reset_index(drop=True)

    series_min = pca_train.min(axis=0)
    series_max = pca_train.max(axis=0)
    pca_train = 2 * (pca_train - series_min) / (series_max - series_min) - 1
    pca_dev = 2 * (pca_dev - series_min) / (series_max - series_min) - 1
    pca_test = 2 * (pca_test - series_min) / (series_max - series_min) - 1


    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])
    series_min = pd.DataFrame(series_min, columns=['series_min'])
    # Merge max serie and min serie
    normalize_indicators = pd.concat([series_max, series_min], axis=1)
    if decomposer=='wd':
        if isinstance(n_components,str):
            pca_data_path = root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'_with_pca_'+n_components+'/'
        else:
            pca_data_path = root_path+'/'+SIGNALS+'/data/'+wavelet_level+'/one_step_1_month_'+PREDICT_PATTERN+'_with_pca_'+str(n_components)+'/'
    else:
        if isinstance(n_components,str):
            pca_data_path = root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'_with_pca_'+n_components+'/'
        else:
            pca_data_path = root_path+'/'+SIGNALS+'/data/one_step_1_month_'+PREDICT_PATTERN+'_with_pca_'+str(n_components)+'/'
    if not os.path.exists(pca_data_path):
        os.makedirs(pca_data_path)
    normalize_indicators.to_csv(pca_data_path+'norm_unsample_id.csv')
    pca_train.to_csv(pca_data_path+'minmax_unsample_train.csv',index=None)
    pca_dev.to_csv(pca_data_path+'minmax_unsample_dev.csv',index=None)
    pca_test.to_csv(pca_data_path+'minmax_unsample_test.csv',index=None)