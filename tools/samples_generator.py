import pandas as pd
import numpy as np
from sklearn import decomposition
import deprecated
import glob
import sys
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
print(10*'-'+' root Path: {}'.format(root_path))


def generate_samples(
    source_path,
    lag,
    column,
    save_path,
    test_len,
    normalizer="max_min",
    seed=None,
    sampling=False,
    header=True,
    index=False,
    ):
    """Generate learning samples for autoregression problem using original time series. 

    Args:

    'source_path' -- ['String'] The source data file path.

    'lag' -- ['int'] The lagged time for original time series.

    'column' -- ['String']The column's name for read the source data by pandas.

    'save_path' --['String'] The path to restore the training, development and testing samples.

    'test_len' --['int'] The length of development and testing set.

    'normalizer' --['string'] Choose which way to normalize the smaples (default 'max_min').

    'seed' --['int'] The seed for sampling (default None).

    'sampling' -- ['Boolean'] Decide wether or not sampling (default True).

    'header' -- ['Boolean'] Decide wether or not save with header (default True).

    'index' -- ['Boolean'] Decide wether or not save with index (default False).
    """
    print("Generating muliti-step hindcast decomposition-ensemble\n training, development and testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)[column]
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)[column]
    # convert pandas dataframe to numpy array
    nparr = np.array(dataframe)
    # Create an empty pandas Dataframe
    full_samples = pd.DataFrame()
    # Generate input series based on lags and add these series to full dataset
    for i in range(lag):
        x = pd.DataFrame(
            nparr[i:dataframe.shape[0] - (lag - i)],
            columns=['X' + str(i + 1)])['X' + str(i + 1)]
        full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))
    # Generate label data
    label = pd.DataFrame(nparr[lag:], columns=['Y'])['Y']
    # Add labled data to full_data_set
    full_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))
    # Get the length of this series
    series_len = full_samples.shape[0]
    # Get the training and developing set
    train_dev_samples = full_samples[0:(series_len - test_len)]
    # Get the testing set.
    test_samples = full_samples[(series_len - test_len):series_len]
    train_dev_len = train_dev_samples.shape[0]
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        train_samples = train_dev_samples.sample(frac=1-(test_len/train_dev_len), random_state=seed)
        dev_samples = train_dev_samples.drop(train_samples.index)
    else:
        train_samples = full_samples[0:(series_len - test_len - test_len)]
        dev_samples = full_samples[(series_len - test_len - test_len):(series_len - test_len)]
    assert (train_samples.shape[0] + dev_samples.shape[0] +test_samples.shape[0]) == series_len
    # Get the max and min value of each series
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    print('Series length:{}'.format(series_len))
    print('Save path:{}'.format(save_path))
    print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))
    if header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+'minmax_sample_train.csv')
        dev_samples.to_csv(save_path+'minmax_sample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+'minmax_sample_train.csv', header=None)
        dev_samples.to_csv(save_path+'minmax_sample_dev.csv', header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', header=None)
    elif not header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+'minmax_unsample_train.csv', header=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', header=None)
    elif not index and header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv( save_path+'minmax_sample_train.csv', index=None)
        dev_samples.to_csv( save_path+'minmax_sample_dev.csv', index=None)
        test_samples.to_csv( save_path+'minmax_unsample_test.csv', index=None)
    elif not index and header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv( save_path+'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv( save_path+'minmax_unsample_dev.csv', index=None)
        test_samples.to_csv( save_path+'minmax_unsample_test.csv', index=None)
    elif not index and not header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+'minmax_sample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_sample_dev.csv', header=None, index=None)
        test_samples.to_csv( save_path+'minmax_unsample_test.csv', header=None, index=None)
    elif not index and not header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+'minmax_unsample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_unsample_dev.csv', header=None, index=None)
        test_samples.to_csv( save_path+'minmax_unsample_test.csv', header=None, index=None)


def gen_one_step_hindcast_samples(
    station,
    decomposer,
    lags,
    input_columns,
    output_column,
    test_len,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False,
):
    """ 
    Generate one step hindcast decomposition-ensemble learning samples. 
    
    Args:

    'station'-- ['string'] The station where the original time series come from.

    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.

    'lags'-- ['int list'] The lagged time for each subsignal.

    'input_columns'-- ['string list'] The input columns' name used for generating the learning samples.

    'output_columns'-- ['string'] The output column's name used for generating the learning samples.

    'test_len'-- ['int'] The size of development and testing samples ().

    'normalizer'-- ['string'] Choose which way to normalize the samples (default 'max_min').

    'seed'-- ['int'] The random seed for sampling the development samples (default None).

    'sampling' -- ['Boolean'] Decide wether or not sampling (default True).

    'header' -- ['Boolean'] Decide wether or not save with header (default True).

    'index' -- ['Boolean'] Decide wether or not save with index (default False).
    """
    print("Generating one step hindcast decomposition-ensemble \ntraining, development and testing samples ...")
    #  Load data from local dick
    if decomposer =="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_1_month_hindcast/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    decompose_file = data_path+decomposer.upper()+"_FULL.csv"
    decompositions = pd.read_csv(decompose_file)
    # Drop NaN
    decompositions.dropna()
    # Get the input data (the decompositions)
    input_data = decompositions[input_columns]
    # Get the output data (the original time series)
    output_data = decompositions[output_column]
    # Get the number of input features
    subsignals_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each subsignal
    full_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one subsignal
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            x = x.reset_index(drop=True)
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
        # make all sample size of each subsignal identical
        oness = oness.iloc[oness.shape[0]-samples_size:]
        oness = oness.reset_index(drop=True)
        full_samples = pd.DataFrame(pd.concat([full_samples,oness],axis=1))
    # Get the target
    target = (output_data.values)[max(lags):]
    target = pd.DataFrame(target,columns=['Y'])
    # Concat the features and target
    full_samples = pd.concat([full_samples,target],axis=1)
    full_samples = pd.DataFrame(full_samples.values,columns=samples_cols)
    full_samples.to_csv(save_path+'full_samples.csv')
    assert samples_size == full_samples.shape[0]
    # Get the training and developing set
    train_dev_samples = full_samples[0:(samples_size - test_len)]
    # Get the testing set.
    test_samples = full_samples[(samples_size - test_len):samples_size]
    train_dev_len = train_dev_samples.shape[0]
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        train_samples = train_dev_samples.sample(frac=1-(test_len/train_dev_len), random_state=seed)
        dev_samples = train_dev_samples.drop(train_samples.index)
    else:
        train_samples = full_samples[0:(samples_size - test_len - test_len)]
        dev_samples = full_samples[(samples_size - test_len - test_len):(samples_size - test_len)]
    assert (train_samples['X1'].size + dev_samples['X1'].size +test_samples['X1'].size) == samples_size
    # Get the max and min value of training set
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('Series length:{}'.format(samples_size))
    print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))
    if  sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv')
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv')
    elif not sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv')
    elif not header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv( save_path+'minmax_sample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv', header=None)
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv', header=None)
    elif not header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv( save_path+'minmax_unsample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv', header=None)
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv', header=None)
    elif not index and header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv', index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)
    elif not index and header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv', index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)
    elif not index and not header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_sample_dev.csv', header=None, index=None)
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv', header=None, index=None)
    elif not index and not header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_unsample_dev.csv', header=None, index=None)
        test_samples.to_csv(save_path+ 'minmax_unsample_test.csv', header=None, index=None)


def gen_one_step_forecast_samples(
    station,
    decomposer,
    lags,
    input_columns,
    output_column,
    start,
    stop,
    test_len,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False,
):
    """ 
    Generate one step forecast decomposition-ensemble samples. 
    
    Args:

    'station'-- ['string'] The station where the original time series come from.

    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.

    'lags'-- ['int list'] The lagged time for subsignals.

    'input_columns'-- ['string lsit'] the input columns' name for read the source data by pandas.

    'output_columns'-- ['string'] the output column's name for read the source data by pandas.

    'start'-- ['int'] The start index of appended decomposition file.

    'stop'-- ['int'] The stop index of appended decomposotion file.
    
    'test_len'-- ['int'] The size of development and testing samples.

    'normalizer'-- ['string'] Choose which way to normalize the samples (default 'max_min').

    'seed'-- ['int'] The seed for sampling (default None).

    'header'-- ['Boolean'] Decide wether or not save with header (default True).

    'index'-- ['Boolean'] Decide wether or not save with index (default False).
    
    """
    print("Generateing one step forecast decomposition-ensemble samples ...")
    #  Load data from local dick
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_1_month_forecast/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # !!!!!!Generate training samples
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions = pd.read_csv(train_decompose_file)
    # Drop NaN
    train_decompositions.dropna()
    # Get the input data (the decompositions)
    train_input_data = train_decompositions[input_columns]
    # Get the output data (the original time series)
    train_output_data = train_decompositions[output_column]
    # Get the number of input features
    subsignals_num = train_input_data.shape[1]
    # Get the data size
    train_data_size = train_input_data.shape[0]
    # Compute the samples size
    train_samples_size = train_data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    train_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one input feature
        one_in = (train_input_data[input_columns[i]]).values #subsignal
        oness = pd.DataFrame() #restor input features
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:train_data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            x = x.reset_index(drop=True)
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
        oness = oness.iloc[oness.shape[0]-train_samples_size:] 
        oness = oness.reset_index(drop=True)
        train_samples = pd.DataFrame(pd.concat([train_samples,oness],axis=1))
    # Get the target
    target = (train_output_data.values)[max(lags):]
    target = pd.DataFrame(target,columns=['Y'])
    # Concat the features and target
    train_samples = pd.concat([train_samples,target],axis=1)
    train_samples = pd.DataFrame(train_samples.values,columns=samples_cols)
    train_samples.to_csv(save_path+'train_samples.csv')
    assert train_samples_size == train_samples.shape[0]
    # normalize the train_samples
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    # !!!!!!!!!!!Generate development and testing samples
    dev_test_samples = pd.DataFrame()
    appended_file_path = data_path+decomposer+"-test/"
    for k in range(start,stop+1):
        #  Load data from local dick
        appended_decompositions = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        appended_decompositions.dropna()
        # Get the input data (the decompositions)
        input_data = appended_decompositions[input_columns]
        # Get the output data (the original time series)
        output_data = appended_decompositions[output_column]
        # Get the number of input features
        subsignals_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate input colmuns for each subsignal
        appended_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one subsignal
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                x = x.reset_index(drop=True)
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            appended_samples = pd.DataFrame(pd.concat([appended_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags):]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        appended_samples = pd.concat([appended_samples,target],axis=1)
        appended_samples = pd.DataFrame(appended_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
        dev_test_samples = pd.concat([dev_test_samples,last_sample],axis=0)
    dev_test_samples = dev_test_samples.reset_index(drop=True)
    dev_test_samples.to_csv(save_path+'dev_test_samples.csv')
    dev_test_samples = 2*(dev_test_samples-series_min)/(series_max-series_min)-1
    dev_samples=dev_test_samples.iloc[0:dev_test_samples.shape[0]-test_len]
    test_samples=dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]

    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None,index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)

def gen_one_step_forecast_samples_triandev_test(
    station,
    decomposer,
    lags,
    input_columns,
    output_column,
    start,
    stop,
    test_len,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False,
):
    """ 
    Generate one step forecast decomposition-ensemble samples. 
    
    Args:

    'station'-- ['string'] The station where the original time series come from.

    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.

    'lags'-- ['int list'] The lagged time for subsignals.

    'input_columns'-- ['string lsit'] the input columns' name for read the source data by pandas.

    'output_columns'-- ['string'] the output column's name for read the source data by pandas.

    'start'-- ['int'] The start index of appended decomposition file.

    'stop'-- ['int'] The stop index of appended decomposotion file.
    
    'test_len'-- ['int'] The size of development and testing samples.

    'normalizer'-- ['string'] Choose which way to normalize the samples (default 'max_min').

    'seed'-- ['int'] The seed for sampling (default None).

    'header'-- ['Boolean'] Decide wether or not save with header (default True).

    'index'-- ['Boolean'] Decide wether or not save with index (default False).
    
    """
    print("Generateing one step forecast decomposition-ensemble samples ...")
    #  Load data from local dick
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_1_month_forecast_traindev_test/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # !!!!!!Generate training samples
    traindev_decompose_file = data_path+decomposer.upper()+"_TRAINDEV.csv"
    traindev_decompositions = pd.read_csv(traindev_decompose_file)
    # Drop NaN
    traindev_decompositions.dropna()
    # Get the input data (the decompositions)
    traindev_input_data = traindev_decompositions[input_columns]
    # Get the output data (the original time series)
    traindev_output_data = traindev_decompositions[output_column]
    # Get the number of input features
    subsignals_num = traindev_input_data.shape[1]
    # Get the data size
    traindev_data_size = traindev_input_data.shape[0]
    # Compute the samples size
    traindev_samples_size = traindev_data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    tran_dev_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one input feature
        one_in = (traindev_input_data[input_columns[i]]).values #subsignal
        oness = pd.DataFrame() #restor input features
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:traindev_data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            x = x.reset_index(drop=True)
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
        oness = oness.iloc[oness.shape[0]-traindev_samples_size:] 
        oness = oness.reset_index(drop=True)
        tran_dev_samples = pd.DataFrame(pd.concat([tran_dev_samples,oness],axis=1))
    # Get the target
    target = (traindev_output_data.values)[max(lags):]
    target = pd.DataFrame(target,columns=['Y'])
    # Concat the features and target
    tran_dev_samples = pd.concat([tran_dev_samples,target],axis=1)
    tran_dev_samples = pd.DataFrame(tran_dev_samples.values,columns=samples_cols)
    tran_dev_samples.to_csv(save_path+'tran_dev_samples.csv')
    train_samples=tran_dev_samples[:tran_dev_samples.shape[0]-120]
    dev_samples = tran_dev_samples[tran_dev_samples.shape[0]-120:]
    assert traindev_samples_size == tran_dev_samples.shape[0]
    # normalize the train_samples
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    # !!!!!!!!!!!Generate testing samples
    
    test_samples = pd.DataFrame()
    appended_file_path = data_path+decomposer+"-test/"
    for k in range(start,stop+1):
        #  Load data from local dick
        appended_decompositions = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        appended_decompositions.dropna()
        # Get the input data (the decompositions)
        input_data = appended_decompositions[input_columns]
        # Get the output data (the original time series)
        output_data = appended_decompositions[output_column]
        # Get the number of input features
        subsignals_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate input colmuns for each subsignal
        appended_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one subsignal
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                x = x.reset_index(drop=True)
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            appended_samples = pd.DataFrame(pd.concat([appended_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags):]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        appended_samples = pd.concat([appended_samples,target],axis=1)
        appended_samples = pd.DataFrame(appended_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
        test_samples = pd.concat([test_samples,last_sample],axis=0)
    test_samples = test_samples.reset_index(drop=True)
    test_samples.to_csv(save_path+'test_samples.csv')
    test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
    assert test_len==test_samples.shape[0]
    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None,index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)


def gen_one_step_forecast_samples_leading_time(
    station,
    decomposer,
    lags,
    input_columns,
    output_column,
    start,
    stop,
    test_len,
    leading_time,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False,
):
    """ 
    Generate one step forecast decomposition-ensemble samples. 
    
    Args:

    'station'-- ['string'] The station where the original time series come from.

    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.

    'lags'-- ['int list'] The lagged time for subsignals.

    'input_columns'-- ['string lsit'] the input columns' name for read the source data by pandas.

    'output_columns'-- ['string'] the output column's name for read the source data by pandas.

    'start'-- ['int'] The start index of appended decomposition file.

    'stop'-- ['int'] The stop index of appended decomposotion file.
    
    'test_len'-- ['int'] The size of development and testing samples.

    'normalizer'-- ['string'] Choose which way to normalize the samples (default 'max_min').

    'seed'-- ['int'] The seed for sampling (default None).

    'header'-- ['Boolean'] Decide wether or not save with header (default True).

    'index'-- ['Boolean'] Decide wether or not save with index (default False).
    
    """
    print("Generateing one step forecast decomposition-ensemble samples ...")
    #  Load data from local dick
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_"+str(leading_time)+"_month_forecast_new/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # !!!!!!Generate training samples
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions = pd.read_csv(train_decompose_file)
    # Drop NaN
    train_decompositions.dropna()
    # Get the input data (the decompositions)
    train_input_data = train_decompositions[input_columns]
    # Get the output data (the original time series)
    train_output_data = train_decompositions[output_column]
    # Get the number of input features
    subsignals_num = train_input_data.shape[1]
    # Get the data size
    train_data_size = train_input_data.shape[0]
    # Compute the samples size
    train_samples_size = train_data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    train_samples = pd.DataFrame()
    for i in range(subsignals_num):
        # Get one input feature
        one_in = (train_input_data[input_columns[i]]).values #subsignal
        oness = pd.DataFrame() #restor input features
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:train_data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            x = x.reset_index(drop=True)
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
        oness = oness.iloc[oness.shape[0]-train_samples_size:] 
        oness = oness.reset_index(drop=True)
        train_samples = pd.DataFrame(pd.concat([train_samples,oness],axis=1))
    # Get the target
    target = (train_output_data.values)[max(lags)+leading_time-1:]
    target = pd.DataFrame(target,columns=['Y'])
    print("target:{}".format(target))
    # Concat the features and target
    train_samples=train_samples[:train_samples.shape[0]-(leading_time-1)]
    train_samples = pd.concat([train_samples,target],axis=1)
    train_samples = pd.DataFrame(train_samples.values,columns=samples_cols)
    train_samples.to_csv(save_path+'train_samples.csv')
    # assert train_samples_size == train_samples.shape[0]
    # normalize the train_samples
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    # !!!!!!!!!!!Generate development and testing samples
    dev_test_samples = pd.DataFrame()
    appended_file_path = data_path+decomposer+"-test/"
    for k in range(start,stop+1):
        #  Load data from local dick
        appended_decompositions = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        appended_decompositions.dropna()
        # Get the input data (the decompositions)
        input_data = appended_decompositions[input_columns]
        # Get the output data (the original time series)
        output_data = appended_decompositions[output_column]
        # Get the number of input features
        subsignals_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate input colmuns for each subsignal
        appended_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one subsignal
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                x = x.reset_index(drop=True)
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            appended_samples = pd.DataFrame(pd.concat([appended_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags)+leading_time-1:]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        appended_samples=appended_samples[:appended_samples.shape[0]-(leading_time-1)]
        appended_samples = pd.concat([appended_samples,target],axis=1)
        appended_samples = pd.DataFrame(appended_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
        dev_test_samples = pd.concat([dev_test_samples,last_sample],axis=0)
    dev_test_samples = dev_test_samples.reset_index(drop=True)
    dev_test_samples.to_csv(save_path+'dev_test_samples.csv')
    dev_test_samples = 2*(dev_test_samples-series_min)/(series_max-series_min)-1
    dev_samples=dev_test_samples.iloc[0:dev_test_samples.shape[0]-test_len]
    test_samples=dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]

    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('The size of training samples:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None,index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)


def gen_multi_step_hindcast_samples(
    station,
    decomposer,
    lags,
    columns,
    test_len,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False,
    ):
    """ 
    Generate muliti-step learning samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags: The lags for autoregression.
        -columns: the columns' name for read the source data by pandas.
        -save_path: The path to restore the training, development and testing samples.
        -test_len: The length of validation(development or testing) set.
        -seed: The seed for sampling.
        -sampling:Boolean, decide wether or not sampling.
        -header:Boolean, decide wether or not save with header.
        -index:Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step hindcast decomposition-ensemble samples ...")
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_1_month_hindcast/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    decompose_file = data_path+decomposer.upper()+"_FULL.csv"
    decompositions = pd.read_csv(decompose_file)

    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Obtain decomposed sub-signal
        sub_signal = decompositions[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            x = x.reset_index(drop=True)
            full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        label = label.reset_index(drop=True)
        # Add labled data to full_data_set
        full_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))
        # Get the length of this series
        series_len = full_samples.shape[0]
        # Get the training and developing set
        train_dev_samples = full_samples[0:(series_len - test_len)]
        # Get the testing set.
        test_samples = full_samples[(series_len - test_len):series_len]
        train_dev_len = train_dev_samples.shape[0]
        # Do sampling if 'sampling' is True
        if sampling:
            # sampling
            np.random.seed(seed)
            train_samples = train_dev_samples.sample(frac=1-(test_len/train_dev_len), random_state=seed)
            dev_samples = train_dev_samples.drop(train_samples.index)
        else:
            train_samples = full_samples[0:(series_len - test_len - test_len)]
            dev_samples = full_samples[(series_len - test_len - test_len):(series_len - test_len)]

        assert (train_samples.shape[0] + dev_samples.shape[0] +test_samples.shape[0]) == series_len

        # Get the max and min value of each series
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
        dev_samples = 2 * (dev_samples - series_min) / (series_max - series_min) - 1
        test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
        # Generate pandas series for series' mean and standard devation
        series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
        series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
        # Merge max serie and min serie
        normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
        # Storage the normalied indicators to local disk
        # print data set length
        print(25*'-')
        print('Series length:{}'.format(series_len))
        print('Save path:{}'.format(save_path))
        print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
        print('The size of training samples:{}'.format(train_samples.shape[0]))
        print('The size of development samples:{}'.format(dev_samples.shape[0]))
        print('The size of testing samples:{}'.format(test_samples.shape[0]))


        if header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv')
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv')
        elif header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv')
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv')
        elif not header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', header=None)
        elif not header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', header=None)
        elif not index and header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv( save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', index=None)
        elif not index and header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv( save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', index=None)
        elif not index and not header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None, index=None)
            test_samples.to_csv( save_path+'minmax_unsample_test_imf'+str(k)+'.csv', header=None, index=None)
        elif not index and not header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None, index=None)
            test_samples.to_csv( save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', header=None, index=None)



def gen_multi_step_forecast_samples(
    station,
    decomposer,
    lags,
    columns,
    start,
    stop,
    test_len,
    normalizer="max_min",
    wavelet_level="db10-lev2",
    seed=None,
    sampling=False,
    header=True,
    index=False
):
    """ 
    Generate multi-step training samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training samples
        -sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n training samples ...")
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_1_month_forcast/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Save path:{}".format(save_path))
    # !!!!!!!!!!Generate training samples
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions = pd.read_csv(train_decompose_file)
    train_decompositions.dropna()
    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Generate sample columns
        samples_columns=[]
        for l in range(1,lags[k]+1):
            samples_columns.append('X'+str(l))
        samples_columns.append('Y')
        # Obtain decomposed sub-signal
        sub_signal = train_decompositions[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        train_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            x = x.reset_index(drop=True)
            train_samples = pd.DataFrame(pd.concat([train_samples, x], axis=1))
        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        label = label.reset_index(drop=True)
        # Add labled data to full_data_set
        train_samples = pd.DataFrame(pd.concat([train_samples, label], axis=1))
        # Do sampling if 'sampling' is True
        if sampling:
            # sampling
            np.random.seed(seed)
            train_samples= train_samples.sample(frac=1, random_state=seed)
        # Get the max and min value of each series
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
        # Generate pandas series for series' mean and standard devation
        series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
        series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
        # Merge max serie and min serie
        normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
        normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
        # print data set length
        print(25*'-')
        print("The size of training samples :{}".format(train_samples.shape[0]))

        # !!!!!Generate development and testing samples
        dev_test_samples = pd.DataFrame()
        appended_file_path = data_path+decomposer+"-test/"
        for j in range(start,stop+1):#遍历每一个附加分解结果
            data = pd.read_csv(appended_file_path+decomposer+'_appended_test'+str(j)+'.csv')
            imf = data[columns[k]]
            nparr = np.array(imf)
            inputs = pd.DataFrame()
            for i in range(lags[k]):
                x = pd.DataFrame(
                    nparr[i:nparr.size - (lags[k] - i)],
                    columns=['X' + str(i + 1)])['X' + str(i + 1)]
                x = x.reset_index(drop=True)
                inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))
            label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
            label=label.reset_index(drop=True)
            full_data_set = pd.DataFrame(pd.concat([inputs, label], axis=1))
            last_imf = full_data_set.iloc[full_data_set.shape[0]-1:]
            dev_test_samples = pd.concat([dev_test_samples,last_imf],axis=0)
        dev_test_samples = dev_test_samples.reset_index(drop=True)
        dev_test_samples = 2*(dev_test_samples-series_min)/(series_max-series_min)-1
        dev_samples = dev_test_samples.iloc[0:dev_test_samples.shape[0]-test_len]
        test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
        dev_samples = dev_samples.reset_index(drop=True)
        test_samples = test_samples.reset_index(drop=True)
        if header and index and sampling:
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv')
            test_samples.to_csv(save_path+'minmax_sample_test_imf'+str(k+1)+'.csv')
        elif header and index and not sampling:
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv')
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv')
        elif not header and index and sampling:
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None)
            test_samples.to_csv(save_path+'minmax_sample_test_imf'+str(k+1)+'.csv', header=None)
        elif not header and index and not sampling:
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', header=None)
        elif not index and header and sampling:
            train_samples.to_csv( save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv( save_path+'minmax_sample_test_imf'+str(k+1)+'.csv', index=None)
        elif not index and header and not sampling:
            train_samples.to_csv( save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv( save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', index=None)
        elif not index and not header and sampling:
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None, index=None)
            test_samples.to_csv(save_path+'minmax_sample_test_imf'+str(k+1)+'.csv', header=None, index=None)
        elif not index and not header and not sampling:
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None, index=None)
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', header=None, index=None)


def gen_one_step_long_leading_forecast_samples(
    station,
    decomposer,
    leading_time,
    input_columns,
    start,
    stop,
    test_len,
    normalizer="max_min",
    pre_times=20,
    filter_boundary=0.2,
    output_clolumn="ORIG",
    wavelet_level="db10-lev2",
    header=True,
    index=False,
):  
    """
    Generate forecast samples for ling leading times.

    """
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_"+str(leading_time)+"_month_forecast_pre"+str(pre_times)+"_thresh"+str(filter_boundary)+"/"
    # files=os.listdir(save_path)
    # if len(files)>0:
    #     print("Samples already generated")
    #     sys.exit(0)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # lag pre_times+leading_time(e.g.,30+3)
    lag=pre_times+leading_time
    pre_cols=[]
    for i in range(1,pre_times+1):
        pre_cols.append("X"+str(i))
    print("Previous columns of lagged months:\n{}".format(pre_cols))
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions= pd.read_csv(train_decompose_file)
    orig = train_decompositions[output_clolumn][lag:]
    orig=orig.reset_index(drop=True)
    selected = {}
    input_df = pd.DataFrame()
    for col in input_columns:
        print("Perform subseries:{}".format(col))
        subsignal = np.array(train_decompositions[col])
        inputs = pd.DataFrame()
        for k in range(lag):
            x = pd.DataFrame(subsignal[k:subsignal.size-(lag-k)],columns=["X"+str(k+1)])["X"+str(k+1)]
            x=x.reset_index(drop=True)
            inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))
        pre_inputs=inputs[pre_cols]
        print("Previous inputs:\n{}".format(pre_inputs.head()))
        partin_out = pd.concat([pre_inputs,orig],axis=1)
        print("Partial inputs and output:\n{}".format(partin_out.head()))
        corrs=partin_out.corr(method="pearson")
        # print("Entire pearson coefficients:\n{}".format(corrs))
        corrs = (corrs[output_clolumn]).iloc[0:corrs.shape[0]-1]
        print("Selected pearson coefficients:\n{}".format(corrs))
        bools = abs(corrs)>=filter_boundary
        print("Conditions judge:{}".format(bools))
        select=list((corrs.loc[bools==True]).index.values)
        print("Selected inputs:\n{}".format(select))
        selected[col]=select
        input_df = pd.concat([input_df,pre_inputs[select]],axis=1)
    print("Selected inputs:\n{}".format(selected))
    print("Entire inputs:\n{}".format(input_df.head()))
    columns = []
    for i in range(0,input_df.shape[1]):
        columns.append("X"+str(i+1))
    columns.append("Y")

    train_samples = pd.DataFrame((pd.concat([input_df,orig],axis=1)).values,columns=columns)
    # normalize the train_samples
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))

    dev_test_samples = pd.DataFrame()
    for i in range(start,stop+1):
        append_decompositions = pd.read_csv(data_path+decomposer+"-test/"+decomposer+"_appended_test"+str(i)+".csv")
        append_orig = append_decompositions[output_clolumn][lag:]
        append_orig = append_orig.reset_index(drop=True)
        append_input_df = pd.DataFrame()
        for col in input_columns:
            append_subsignal = np.array(append_decompositions[col])
            append_inputs=pd.DataFrame()
            for k in range(lag):
                x = pd.DataFrame(append_subsignal[k:append_subsignal.size-(lag-k)],columns=["X"+str(k+1)])["X"+str(k+1)]
                x = x.reset_index(drop=True)
                append_inputs=pd.concat([append_inputs,x],axis=1)
            append_input_df = pd.concat([append_input_df,append_inputs[selected[col]]],axis=1)
        append_samples=pd.concat([append_input_df,append_orig],axis=1)
        append_samples = pd.DataFrame(append_samples.values,columns=columns)
        # print(append_samples.tail())
        last_sample = append_samples.iloc[append_samples.shape[0]-1:]
        dev_test_samples=pd.concat([dev_test_samples,last_sample],axis=0)
    # print(dev_test_samples)
    dev_test_samples=dev_test_samples.reset_index(drop=True)
    dev_test_samples = 2*(dev_test_samples-series_min)/(series_max-series_min)-1
    dev_samples = dev_test_samples.iloc[0:dev_test_samples.shape[0]-test_len]
    test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
    dev_samples = dev_samples.reset_index(drop=True)
    test_samples = test_samples.reset_index(drop=True)
    
    normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
    if header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        train_samples.to_csv(save_path+'minmax_unsample_train.csv',header=None,index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None) 


if __name__ == "__main__":

    leading_times = [3,5,7,9]
    filter_boundary=0.1

    # gen_one_step_forecast_samples(
    #     station="Huaxian",
    #     decomposer="eemd",

    # )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Huaxian",
    #         decomposer="eemd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9',],
    #         start = 553,
    #         stop= 792,#792
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Huaxian",
    #         decomposer="vmd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Huaxian",
    #         decomposer="ssa",
    #         leading_time=leading_time,
    #         input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Huaxian",
    #         decomposer="wd",
    #         leading_time=leading_time,
    #         input_columns=['D1','D2','A2',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Xianyang",
    #         decomposer="eemd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9',],
    #         start = 553,
    #         stop= 792,#792
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Xianyang",
    #         decomposer="vmd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Xianyang",
    #         decomposer="ssa",
    #         leading_time=leading_time,
    #         input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Xianyang",
    #         decomposer="wd",
    #         leading_time=leading_time,
    #         input_columns=['D1','D2','A2',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Zhangjiashan",
    #         decomposer="eemd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9',],
    #         start = 553,
    #         stop= 792,#792
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Zhangjiashan",
    #         decomposer="vmd",
    #         leading_time=leading_time,
    #         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Zhangjiashan",
    #         decomposer="ssa",
    #         leading_time=leading_time,
    #         input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    # for leading_time in leading_times:
    #     gen_one_step_long_leading_forecast_samples(
    #         station="Zhangjiashan",
    #         decomposer="wd",
    #         leading_time=leading_time,
    #         input_columns=['D1','D2','A2',],
    #         start=553,
    #         stop=792,
    #         test_len=120,
    #         filter_boundary=filter_boundary,
    #     )

    