import pandas as pd
import numpy as np
from sklearn import decomposition
import deprecated
import glob
import sys
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
parent_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
print(10*'-'+' Current Path: {}'.format(root_path))
print(10*'-'+' Parent Path: {}'.format(parent_path))
print(10*'-'+' Grandpa Path: {}'.format(grandpa_path))

def gen_samples_minmax(
    source_path,
    lag,
    column,
    save_path,
    val_len,
    seed=None,
    sampling=False,
    header=True,
    index=False,
    ):
    """ 
    Generate muliti-step learning samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the learning data.
        -lag: The lags for autoregression.
        -column: the column's name for read the source data by pandas.
        -save_path: The path to restore the training, development and testing samples.
        -val_len: The length of validation(development or testing) set.
        -seed: The seed for sampling.
        -sampling:Boolean, decide wether or not sampling.
        -header:Boolean, decide wether or not save with header.
        -index:Boolean, decide wether or not save with index.
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
    train_dev_samples = full_samples[0:(series_len - val_len)]
    # Get the testing set.
    test_samples = full_samples[(series_len - val_len):series_len]
    train_dev_len = train_dev_samples.shape[0]
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        train_samples = train_dev_samples.sample(frac=1-(val_len/train_dev_len), random_state=seed)
        dev_samples = train_dev_samples.drop(train_samples.index)
    else:
        train_samples = full_samples[0:(series_len - val_len - val_len)]
        dev_samples = full_samples[(series_len - val_len - val_len):(series_len - val_len)]

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

# For one step hindcast
def gen_one_step_samples_minmax(
                    source_path,
                    lags,
                    input_columns,
                    output_column,
                    save_path,
                    val_len=None,
                    seed=None,
                    sampling=False,
                    header=True,
                    index=False,
                ):
    """ 
    Generate one step hindcast decomposition-ensemble learning samples
    (training, development and testing) for autoregression problem. 
    This program could generate source CSV flie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the learning data.
        -lags: The lags for autoregression.
        -input_columns: The input columns used for generating the learning samples.
        -output_columns: The output columns used for generating the learning samples.
        -svae_path: The save path for svaing the training, development, testing samples and normalized indicators.
        -val_len: The size of validation samples.
        -seed: The random seed for sampling the development samples.
        -sampling: True if sample the development samples from training-development samples.
        -header: True if save the training, development and testing samples from full samples.
        -sampling: True if save the training, development and testing samples from full samples.
    
    """
    print("Generating one step hindcast decomposition-ensemble \ntraining, development and testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()
    # Get the input data (the decompositions)
    input_data = dataframe[input_columns]
    # Get the output data (the original time series)
    output_data = dataframe[output_column]
    # Get the number of input features
    features_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    full_samples = pd.DataFrame()
    for i in range(features_num):
        # Get one input feature
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
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
    train_dev_samples = full_samples[0:(samples_size - val_len)]
    # Get the testing set.
    test_samples = full_samples[(samples_size - val_len):samples_size]
    train_dev_len = train_dev_samples.shape[0]
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        train_samples = train_dev_samples.sample(frac=1-(val_len/train_dev_len), random_state=seed)
        dev_samples = train_dev_samples.drop(train_samples.index)
    else:
        train_samples = full_samples[0:(samples_size - val_len - val_len)]
        dev_samples = full_samples[(samples_size - val_len - val_len):(samples_size - val_len)]

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

# For one step hindcast



# for one step forecast (train and dev)
def gen_one_step_train_dev_samples_minmax(
                source_path,
                lags,
                input_columns,
                output_column,
                save_path,
                dev_len,
                seed=None,
                sampling=False,
                header=True,
                index=False,
            ):
    """ 
    Generate one step decomposition-ensemble training and development samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    \nArgs:
        \n-source_path: The source data file path for generate the training and development samples.
        \n-lags: The lags for autoregression.
        \n-input_columns: the input columns' name for read the source data by pandas.
        \n-output_columns: the output column's name for read the source data by pandas.
        \n-save_path: The path of restoring the training and development samples.
        \n-dev_len: The size of development samples.
        \n-seed: The seed for sampling.
        \n-sampling: Boolean, decide wether or not sampling.
        \n-header: Boolean, decide wether or not save with header.
        \n-index: Boolean, Decide wether or not save with index.
    
    """
    print("Generateing one step forecast decomposition-ensemble\n training and development samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()
    # Get the input data (the decompositions)
    input_data = dataframe[input_columns]
    # Get the output data (the original time series)
    output_data = dataframe[output_column]
    # Get the number of input features
    features_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    full_samples = pd.DataFrame()
    for i in range(features_num):
        # Get one input feature
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
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
    train_dev_samples = full_samples
    # Get the size of training-development samples.
    train_dev_len = train_dev_samples.shape[0]
    assert train_dev_len == samples_size
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        dev_frac=(dev_len/train_dev_len)
        dev_samples= train_dev_samples.sample(frac=dev_frac, random_state=seed)
        train_samples = train_dev_samples.drop(dev_samples.index)
    else:
        train_samples = train_dev_samples[0:(train_dev_len - dev_len)]
        dev_samples = train_dev_samples[(train_dev_len - dev_len):train_dev_len]

    assert (train_samples['X1'].size + dev_samples['X1'].size) == samples_size
    
    # Get the max and min value of training set
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
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    
    print('Save path:{}'.format(save_path))
    print('Samples size:{}'.format(samples_size))
    print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
    print('The size of training samples size:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))

    if  sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv')
    elif not sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv')
    elif not header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv( save_path+'minmax_sample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv', header=None)
    elif not header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv( save_path+'minmax_unsample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv', header=None)
    elif not index and header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev.csv', index=None)
    elif not index and header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev.csv', index=None)
    elif not index and not header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_sample_dev.csv', header=None, index=None)
    elif not index and not header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_unsample_dev.csv', header=None, index=None)




def gen_one_step_train_dev1_samples_minmax(
                source_path,
                lags,
                input_columns,
                output_column,
                save_path,
                dev_len,
                seed=None,
                sampling=False,
                header=True,
                index=False,
            ):
    """ 
    Generate one step decomposition-ensemble training and development samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the training and development samples.
        -lags: The lags for autoregression.
        -input_columns: the input columns' name for read the source data by pandas.
        -output_columns: the output column's name for read the source data by pandas.
        -save_path: The path of restoring the training and development samples.
        -dev_len: The size of development samples.
        -seed: The seed for sampling.
        -sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, Decide wether or not save with index.
    
    """
    print("Generateing one step forecast decomposition-ensemble\n training and development(1) samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()
    # Get the input data (the decompositions)
    input_data = dataframe[input_columns]
    # Get the output data (the original time series)
    output_data = dataframe[output_column]
    # Get the number of input features
    features_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    full_samples = pd.DataFrame()
    for i in range(features_num):
        # Get one input feature
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
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
    train_dev_samples = full_samples
    # Get the size of training-development samples.
    train_dev_len = train_dev_samples.shape[0]
    assert train_dev_len == samples_size
    # Do sampling if 'sampling' is True
    if sampling:
        # sampling
        np.random.seed(seed)
        dev_frac=(dev_len/train_dev_len)
        dev_samples= train_dev_samples.sample(frac=dev_frac, random_state=seed)
        train_samples = train_dev_samples.drop(dev_samples.index)
    else:
        train_samples = train_dev_samples[0:(train_dev_len - dev_len)]
        dev_samples = train_dev_samples[(train_dev_len - dev_len):train_dev_len]

    assert (train_samples['X1'].size + dev_samples['X1'].size) == samples_size
    
    # Get the max and min value of training set
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
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    
    print('Save path:{}'.format(save_path))
    print('Samples size:{}'.format(samples_size))
    print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
    print('The size of training samples size:{}'.format(train_samples.shape[0]))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))

    if  sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_sample_dev1.csv')
    elif not sampling and (header and index):
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv')
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev1.csv')
    elif not header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv( save_path+'minmax_sample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev1.csv', header=None)
    elif not header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv( save_path+'minmax_unsample_train.csv', header=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev1.csv', header=None)
    elif not index and header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_sample_dev1.csv', index=None)
    elif not index and header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path+ 'minmax_unsample_dev1.csv', index=None)
    elif not index and not header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_sample_dev1.csv', header=None, index=None)
    elif not index and not header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', header=None, index=None)
        dev_samples.to_csv( save_path+'minmax_unsample_dev1.csv', header=None, index=None)



def gen_one_step_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    input_columns,
    output_column,
    save_path,
    start,
    stop,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    Generate one step decomposition-ensemble testing samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    \nArgs:
        \n-source_path: The source path of appended testing decomposition files for generate the testing samples.
        \n-decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        \n-lags: The lags for autoregression.
        \n-input_columns: the input columns' name for read the source data by pandas.
        \n-output_columns: the output column's name for read the source data by pandas.
        \n-save_path: The path of restoring the testing samples.
        \n-start: The start index of appended testing decomposition.
        \n-stop: The stop index of appended testing decomposition.
        \n-is_sampled: Boolean, wether or not the training samples have been sampled.
        \n-header: Boolean, decide wether or not save with header.
        \n-index: Boolean, Decide wether or not save with index.
    """
    print("Generating one step forecast decomposition-ensemble\n testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test_samples = pd.DataFrame()
    for k in range(start,stop+1):
        #  Load data from local dick
        dataframe = pd.read_csv(source_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        dataframe.dropna()
        # Get the input data (the decompositions)
        input_data = dataframe[input_columns]
        # Get the output data (the original time series)
        output_data = dataframe[output_column]
        # Get the number of input features
        features_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate feature columns
        samples_cols = []
        for i in range(sum(lags)):
            samples_cols.append('X'+str(i+1))
        samples_cols.append('Y')
        # Generate input colmuns for each input feature
        full_samples = pd.DataFrame()
        for i in range(features_num):
            # Get one input feature
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            full_samples = pd.DataFrame(pd.concat([full_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags):]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        full_samples = pd.concat([full_samples,target],axis=1)
        full_samples = pd.DataFrame(full_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = full_samples.iloc[full_samples.shape[0]-1:]
        test_samples = pd.concat([test_samples,last_sample],axis=0)
    test_samples.to_csv(save_path+'test_samples.csv')
    if is_sampled:
        norm_id = pd.read_csv(save_path+'norm_sample_id.csv')
    else:
        norm_id = pd.read_csv(save_path+'norm_unsample_id.csv')
    series_max = (norm_id['series_max']).values
    series_min = (norm_id['series_min']).values
    test_samples=test_samples.values
    test_samples = 2 * (test_samples - series_min) / (series_max - series_min) - 1
    test_samples = pd.DataFrame(test_samples,columns=samples_cols)

    print(25*'-')
    
    print('Save path:{}'.format(save_path))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)



# For one step forecast (train)
def gen_one_step_train_samples_minmax(
                source_path,
                lags,
                input_columns,
                output_column,
                save_path,
                seed = None,
                sampling=False,
                header=True,
                index=False,
            ):
    """ 
    Generate one step decomposition-ensemble training samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the training samples.
        -lags: The lags for autoregression.
        -input_columns: the input columns' name for read the source data by pandas.
        -output_columns: the output column's name for read the source data by pandas.
        -save_path: The path of restoring the training samples.
        -seed: The random seed for sampling training samples.
        -Sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, Decide wether or not save with index.
    
    """
    print("Generating one step forecast decomposition-ensemble\n training samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()
    # Get the input data (the decompositions)
    input_data = dataframe[input_columns]
    # Get the output data (the original time series)
    output_data = dataframe[output_column]
    # Get the number of input features
    features_num = input_data.shape[1]
    # Get the data size
    data_size = input_data.shape[0]
    # Compute the samples size
    samples_size = data_size-max(lags)
    # Generate feature columns
    samples_cols = []
    for i in range(sum(lags)):
        samples_cols.append('X'+str(i+1))
    samples_cols.append('Y')
    # Generate input colmuns for each input feature
    full_samples = pd.DataFrame()
    for i in range(features_num):
        # Get one input feature
        one_in = (input_data[input_columns[i]]).values
        oness = pd.DataFrame()
        for j in range(lags[i]): 
            x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
            oness = pd.DataFrame(pd.concat([oness,x],axis=1))
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

    # Only for training samples
    train_samples = full_samples

    if sampling:
        # sampling
        np.random.seed(seed)
        train_samples = train_samples.sample(frac=1, random_state=seed)

    
    # Get the max and min value of training set
    series_max = train_samples.max(axis=0)
    series_min = train_samples.min(axis=0)
    # Normalize each series to the range between -1 and 1
    train_samples = 2 * (train_samples - series_min) / (series_max - series_min) - 1
    # Generate pandas series for series' mean and standard devation
    series_max = pd.DataFrame(series_max, columns=['series_max'])['series_max']
    series_min = pd.DataFrame(series_min, columns=['series_min'])['series_min']
    # Merge max serie and min serie
    normalize_indicators = pd.DataFrame(pd.concat([series_max, series_min], axis=1))
    # Storage the normalied indicators to local disk
    # print data set length
    print(25*'-')
    
    print('Save path:{}'.format(save_path))
    print('Samples size:{}'.format(samples_size))
    print('The size of training samples size:{}'.format(train_samples.shape[0]))

    if  header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv')
    elif header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv')
    
    elif not header and index and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv( save_path+'minmax_sample_train.csv', header=None)
    
    elif not header and index and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv( save_path+'minmax_unsample_train.csv', header=None)
    
    elif not index and header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', index=None)

    elif not index and header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', index=None)
    
    elif not index and not header and sampling:
        normalize_indicators.to_csv(save_path+'norm_sample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_sample_train.csv', header=None, index=None)

    elif not index and not header and not sampling:
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+ 'minmax_unsample_train.csv', header=None, index=None)

def gen_one_step_dev_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    input_columns,
    output_column,
    save_path,
    start,
    stop,
    test_len,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    \nGenerate one step decomposition-ensemble testing samples for autoregression problem. 
    \nThis program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        \n-source_path: The source path of appended testing decomposition files for generate the testing samples.
        \n-decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        \n-lags: The lags for autoregression.
        \n-input_columns: the input columns' name for read the source data by pandas.
        \n-output_columns: the output column's name for read the source data by pandas.
        \n-save_path: The path of restoring the testing samples.
        \n-start: The start index of appended testing decomposition.
        \n-stop: The stop index of appended testing decomposition.
        \n-test_len: The size of testing samples.
        \n-is_sampled: Boolean, wether or not the training samples have been sampled.
        \n-header: Boolean, decide wether or not save with header.
        \n-index: Boolean, Decide wether or not save with index.
    """
    print("Generating one step forecast decomposition-ensemble\n development and testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dev_test_samples = pd.DataFrame()

    for k in range(start,stop+1):
        #  Load data from local dick
        dataframe = pd.read_csv(source_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        dataframe.dropna()
        # Get the input data (the decompositions)
        input_data = dataframe[input_columns]
        # Get the output data (the original time series)
        output_data = dataframe[output_column]
        # Get the number of input features
        features_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate feature columns
        samples_cols = []
        for i in range(sum(lags)):
            samples_cols.append('X'+str(i+1))
        samples_cols.append('Y')
        # Generate input colmuns for each input feature
        full_samples = pd.DataFrame()
        for i in range(features_num):
            # Get one input feature
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            full_samples = pd.DataFrame(pd.concat([full_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags):]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        full_samples = pd.concat([full_samples,target],axis=1)
        full_samples = pd.DataFrame(full_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = full_samples.iloc[full_samples.shape[0]-1:]
        dev_test_samples = pd.concat([dev_test_samples,last_sample],axis=0)
    dev_test_samples.to_csv(save_path+'dev_test_samples.csv')
    if is_sampled:
        norm_id = pd.read_csv(save_path+'norm_sample_id.csv')
    else:
        norm_id = pd.read_csv(save_path+'norm_unsample_id.csv')
    series_max = (norm_id['series_max']).values
    series_min = (norm_id['series_min']).values
    dev_test_samples=dev_test_samples.values
    dev_test_samples = 2 * (dev_test_samples - series_min) / (series_max - series_min) - 1
    dev_test_samples = pd.DataFrame(dev_test_samples,columns=samples_cols)

    dev_samples=dev_test_samples.iloc[0:test_len]
    test_samples=dev_test_samples.iloc[test_len:]
    print(25*'-')
    
    print('Save path:{}'.format(save_path))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)

def gen_one_step_dev2_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    input_columns,
    output_column,
    save_path,
    start,
    stop,
    test_len,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    \nGenerate one step decomposition-ensemble testing samples for autoregression problem. 
    \nThis program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        \n-source_path: The source path of appended testing decomposition files for generate the testing samples.
        \n-decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        \n-lags: The lags for autoregression.
        \n-input_columns: the input columns' name for read the source data by pandas.
        \n-output_columns: the output column's name for read the source data by pandas.
        \n-save_path: The path of restoring the testing samples.
        \n-start: The start index of appended testing decomposition.
        \n-stop: The stop index of appended testing decomposition.
        \n-test_len: The size of testing samples.
        \n-is_sampled: Boolean, wether or not the training samples have been sampled.
        \n-header: Boolean, decide wether or not save with header.
        \n-index: Boolean, Decide wether or not save with index.
    """
    print("Generating one step forecast decomposition-ensemble\n development(2) and testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    dev_test_samples = pd.DataFrame()

    for k in range(start,stop+1):
        #  Load data from local dick
        dataframe = pd.read_csv(source_path+decomposer+'_appended_test'+str(k)+'.csv')  
        # Drop NaN
        dataframe.dropna()
        # Get the input data (the decompositions)
        input_data = dataframe[input_columns]
        # Get the output data (the original time series)
        output_data = dataframe[output_column]
        # Get the number of input features
        features_num = input_data.shape[1]
        # Get the data size
        data_size = input_data.shape[0]
        # Compute the samples size
        samples_size = data_size-max(lags)
        # Generate feature columns
        samples_cols = []
        for i in range(sum(lags)):
            samples_cols.append('X'+str(i+1))
        samples_cols.append('Y')
        # Generate input colmuns for each input feature
        full_samples = pd.DataFrame()
        for i in range(features_num):
            # Get one input feature
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            for j in range(lags[i]): 
                x = pd.DataFrame(one_in[j:data_size-(lags[i]-j)],columns=['X' + str(j + 1)])['X' + str(j + 1)]
                oness = pd.DataFrame(pd.concat([oness,x],axis=1))
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            full_samples = pd.DataFrame(pd.concat([full_samples,oness],axis=1))
        # Get the target
        target = (output_data.values)[max(lags):]
        target = pd.DataFrame(target,columns=['Y'])
        # Concat the features and target
        full_samples = pd.concat([full_samples,target],axis=1)
        full_samples = pd.DataFrame(full_samples.values,columns=samples_cols)
        # Get the last sample of full samples
        last_sample = full_samples.iloc[full_samples.shape[0]-1:]
        dev_test_samples = pd.concat([dev_test_samples,last_sample],axis=0)
    dev_test_samples.to_csv(save_path+'dev_test_samples.csv')
    if is_sampled:
        norm_id = pd.read_csv(save_path+'norm_sample_id.csv')
    else:
        norm_id = pd.read_csv(save_path+'norm_unsample_id.csv')
    series_max = (norm_id['series_max']).values
    series_min = (norm_id['series_min']).values
    dev_test_samples=dev_test_samples.values
    dev_test_samples = 2 * (dev_test_samples - series_min) / (series_max - series_min) - 1
    dev_test_samples = pd.DataFrame(dev_test_samples,columns=samples_cols)

    dev_samples=dev_test_samples.iloc[0:test_len]
    test_samples=dev_test_samples.iloc[test_len:]
    print(25*'-')
    print('Save path:{}'.format(save_path))
    print('The size of development samples:{}'.format(dev_samples.shape[0]))
    print('The size of testing samples:{}'.format(test_samples.shape[0]))

    if header and index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev2.csv')
        test_samples.to_csv(save_path+'minmax_unsample_test.csv')
    elif not header and index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev2.csv',header=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None)
    elif header and not index:
        dev_samples.to_csv(save_path+'minmax_unsample_dev2.csv',index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',index=None)
    else:
        dev_samples.to_csv(save_path+'minmax_unsample_dev2.csv',header=None,index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv',header=None,index=None)

def gen_multi_step_samples_minmax(
    source_path,
    lags,
    columns,
    save_path,
    val_len,
    seed=None,
    sampling=False,
    header=True,
    index=False,
    ):
    """ 
    Generate muliti-step learning samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the learning data.
        -lags: The lags for autoregression.
        -columns: the columns' name for read the source data by pandas.
        -save_path: The path to restore the training, development and testing samples.
        -val_len: The length of validation(development or testing) set.
        -seed: The seed for sampling.
        -sampling:Boolean, decide wether or not sampling.
        -header:Boolean, decide wether or not save with header.
        -index:Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step hindcast decomposition-ensemble\n training, development and testing samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    #  Load data from local dick
    if '.xlsx' in source_path:
        dataframe = pd.read_excel(source_path)
    elif '.csv' in source_path:
        dataframe = pd.read_csv(source_path)

    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Obtain decomposed sub-signal
        sub_signal = dataframe[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        # Add labled data to full_data_set
        full_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))
        # Get the length of this series
        series_len = full_samples.shape[0]
        # Get the training and developing set
        train_dev_samples = full_samples[0:(series_len - val_len)]
        # Get the testing set.
        test_samples = full_samples[(series_len - val_len):series_len]
        train_dev_len = train_dev_samples.shape[0]
        # Do sampling if 'sampling' is True
        if sampling:
            # sampling
            np.random.seed(seed)
            train_samples = train_dev_samples.sample(frac=1-(val_len/train_dev_len), random_state=seed)
            dev_samples = train_dev_samples.drop(train_samples.index)
        else:
            train_samples = full_samples[0:(series_len - val_len - val_len)]
            dev_samples = full_samples[(series_len - val_len - val_len):(series_len - val_len)]

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

def gen_multi_step_train_dev_samples_minmax(
                    source_path,
                    lags,
                    columns,
                    save_path,
                    dev_len=None,
                    seed=None,
                    sampling=False,
                    header=True,
                    index=False
                    ):
    """ 
    Generate multi-step training and development samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the training and development samples.
        -lags: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training and development samples
        -dev_len: The length of development samples.
        -seed: The seed for sampling.
        -sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n training and development samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if str.find(source_path,'.xlsx')!=-1:
        dataframe = pd.read_excel(source_path)
    if str.find(source_path,'.csv')!=-1:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()

    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Obtain decomposed sub-signal
        sub_signal = dataframe[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        # Add labled data to full_data_set
        train_dev_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))

        # Get the length of this series
        series_len = len(train_dev_samples)

        # Do sampling if 'sampling' is True
        if sampling:
            # sampling
            np.random.seed(seed)
            dev_frac=(dev_len/series_len)
            dev_samples= train_dev_samples.sample(frac=dev_frac, random_state=seed)
            train_samples = train_dev_samples.drop(dev_samples.index)
        else:
            train_samples = train_dev_samples[0:(series_len - dev_len)]
            dev_samples = train_dev_samples[(series_len - dev_len):series_len]

         # Get the max and min value of each series
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
        # Storage the normalied indicators to local disk
        


        # print data set length
        print(25*'-')
        print('Save path:{}'.format(save_path))
        print('Series length:{}'.format(series_len))
        print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
        print('The size of training samples:{}'.format(train_samples.shape[0]))
        print('The size of development samples:{}'.format(dev_samples.shape[0]))


        if header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv')
        elif header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv')
        elif not header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None)
        elif not header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None)
        elif not index and header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_sample_train_imf'+str(k)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev_imf'+str(k)+'.csv', index=None)
        elif not index and header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', index=None)
        elif not index and not header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev_imf'+str(k+1)+'.csv', header=None, index=None)
        elif not index and not header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', header=None, index=None)

def gen_multi_step_train_dev1_samples_minmax(
                    source_path,
                    lags,
                    columns,
                    save_path,
                    dev_len=None,
                    seed=None,
                    sampling=False,
                    header=True,
                    index=False
                    ):
    """ 
    Generate multi-step training and development samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the training and development samples.
        -lags: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training and development samples
        -dev_len: The length of development samples.
        -seed: The seed for sampling.
        -sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n training and development(1) samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if str.find(source_path,'.xlsx')!=-1:
        dataframe = pd.read_excel(source_path)
    if str.find(source_path,'.csv')!=-1:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()

    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Obtain decomposed sub-signal
        sub_signal = dataframe[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        # Add labled data to full_data_set
        train_dev_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))

        # Get the length of this series
        series_len = len(train_dev_samples)

        # Do sampling if 'sampling' is True
        if sampling:
            # sampling
            np.random.seed(seed)
            dev_frac=(dev_len/series_len)
            dev_samples= train_dev_samples.sample(frac=dev_frac, random_state=seed)
            train_samples = train_dev_samples.drop(dev_samples.index)
        else:
            train_samples = train_dev_samples[0:(series_len - dev_len)]
            dev_samples = train_dev_samples[(series_len - dev_len):series_len]

         # Get the max and min value of each series
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
        # Storage the normalied indicators to local disk
        


        # print data set length
        print(25*'-')
        print('Series length:{}'.format(series_len))
        print('Save path:{}'.format(save_path))
        print('The size of training and development samples:{}'.format(train_dev_samples.shape[0]))
        print('The size of training samples:{}'.format(train_samples.shape[0]))
        print('The size of development samples:{}'.format(dev_samples.shape[0]))


        if header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_sample_dev1_imf'+str(k+1)+'.csv')
        elif header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv')
            dev_samples.to_csv(save_path+'minmax_unsample_dev1_imf'+str(k+1)+'.csv')
        elif not header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_sample_dev1_imf'+str(k+1)+'.csv', header=None)
        elif not header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev1_imf'+str(k+1)+'.csv', header=None)
        elif not index and header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev1_imf'+str(k+1)+'.csv', index=None)
        elif not index and header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev1_imf'+str(k+1)+'.csv', index=None)
        elif not index and not header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv( save_path+'minmax_sample_dev1_imf'+str(k+1)+'.csv', header=None, index=None)
        elif not index and not header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None, index=None)
            dev_samples.to_csv( save_path+'minmax_unsample_dev1_imf'+str(k+1)+'.csv', header=None, index=None)

def gen_multi_step_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    columns,
    save_path,
    start,
    stop,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    Generate multi-step decomposition-ensemble testing samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source path of appended testing decomposition files for generate the testing samples.
        -decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        -lags: The lags for autoregression, e.g., [1,2,3,4,5].
        -columns: the columns' name for read the source data by pandas, e.g., ['a','b','c','d','e'].
        -save_path: The path of restoring the testing samples.
        -start: The start index of appended testing decomposition.
        -stop: The stop index of appended testing decomposition.
        -is_sampled: Boolean, wether or not the training samples have beed sampled.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, Decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n testing samples ...")
    for i in range(1,len(lags)+1):#imfs
        lag=lags[i-1]#获取该IMF的滞后天数（输入变量）
        if lag==0:
            print("The lag of sub-signal({:.0f})".format(i)+" equals to 0")
            continue
        samples_columns=[]
        for l in range(1,lag+1):
            samples_columns.append('X'+str(l))
        samples_columns.append('Y')
        test_imf_df = pd.DataFrame()#测试集样本:输入为分解序列，输出为分解序列
        for j in range(start,stop+1):#遍历每一个附加分解结果
            data = pd.read_csv(source_path+decomposer+'_appended_test'+str(j)+'.csv')#取出逐天将test数据附加到traindev后的附加数据集分解的结果
            imf = data[columns[i-1]]#取出这一天之前相应的IMF
            nparr = np.array(imf)#转换为numpy array
            inputs = pd.DataFrame()#这一天之前分解结果形成的训练样本的输入
            for k in range(lag):#逐个生成输入特征变量
                x = pd.DataFrame(
                    nparr[k:nparr.size - (lag - k)],
                    columns=['X' + str(k + 1)])['X' + str(k + 1)]
                inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))#合并输入特征变量
            label = pd.DataFrame(nparr[lag:], columns=['Y'])['Y']#生成输出标签列
            full_data_set = pd.DataFrame(pd.concat([inputs, label], axis=1))#合并输入与输出，生成样本
            last_imf = full_data_set.iloc[full_data_set.shape[0]-1:]#取出最后一行样本作为该天的样本
            test_imf_df = pd.concat([test_imf_df,last_imf],axis=0)#合并所有天的样本
        test_imf_df=test_imf_df.reset_index(drop=True)#重新索引测试集样本
        # 用子序列训练集的归一化参数来归一化测试集
        if is_sampled:
            norm_id = pd.read_csv(save_path+'norm_sample_id_imf' + str(i) + '.csv')
        else:
            norm_id = pd.read_csv(save_path+'norm_unsample_id_imf' + str(i) + '.csv')
        series_max = (norm_id['series_max']).values
        series_min = (norm_id['series_min']).values
        test_imf_df=test_imf_df.values
        test_imf_df = 2 * (test_imf_df - series_min) / (series_max - series_min) - 1
        test_imf_df = pd.DataFrame(test_imf_df,columns=samples_columns)
        test_imf_df.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',index=None) 

def gen_multi_step_train_samples_minmax(
                    source_path,
                    lags,
                    columns,
                    save_path,
                    seed=None,
                    sampling=False,
                    header=True,
                    index=False
                    ):
    """ 
    Generate multi-step training samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source data file path for generate the training samples.
        -lags: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training samples
        -sampling: Boolean, decide wether or not sampling.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n training samples ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #  Load data from local dick
    if str.find(source_path,'.xlsx')!=-1:
        dataframe = pd.read_excel(source_path)
    if str.find(source_path,'.csv')!=-1:
        dataframe = pd.read_csv(source_path)
    # Drop NaN
    dataframe.dropna()

    for k in range(len(columns)):
        if lags[k]==0:
            print("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
            continue
        # Obtain decomposed sub-signal
        sub_signal = dataframe[columns[k]]
        # convert pandas dataframe to numpy array
        nparr = np.array(sub_signal)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lags and add these series to full dataset
        for i in range(lags[k]):
            x = pd.DataFrame(
                nparr[i:sub_signal.shape[0] - (lags[k] - i)],
                columns=['X' + str(i + 1)])['X' + str(i + 1)]
            full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

        # Generate label data
        label = pd.DataFrame(nparr[lags[k]:], columns=['Y'])['Y']
        # Add labled data to full_data_set
        train_samples = pd.DataFrame(pd.concat([full_samples, label], axis=1))

        # Get the length of this series
        series_len = len(train_samples)

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
        


        # print data set length
        print(25*'-')
        print('Save path:{}'.format(save_path))
        print('Series length:{}'.format(series_len))
        print('The size of training samples :{}'.format(train_samples.shape[0]))
        


        if header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv')
        elif header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv')
        elif not header and index and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None)
        elif not header and index and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None)
        elif not index and header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', index=None)
        elif not index and header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv( save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
        elif not index and not header and sampling:
            normalize_indicators.to_csv(save_path+'norm_sample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_sample_train_imf'+str(k+1)+'.csv', header=None, index=None)
        elif not index and not header and not sampling:
            normalize_indicators.to_csv(save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', header=None, index=None)

def gen_multi_step_dev_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    columns,
    save_path,
    start,
    stop,
    test_len,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    Generate multi-step decomposition-ensemble testing samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source path of appended testing decomposition files for generate the testing samples.
        -decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        -lags: The lags for autoregression, e.g., [1,2,3,4,5].
        -columns: the columns' name for read the source data by pandas, e.g., ['a','b','c','d','e'].
        -save_path: The path of restoring the testing samples.
        -start: The start index of appended testing decomposition.
        -stop: The stop index of appended testing decomposition.
        -test_len: The length of testing samples.
        -is_sampled: Boolean, wether or not the training samples have been sampled.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, Decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n development and testing samples ...")
    for i in range(1,len(lags)+1):#imfs
        lag=lags[i-1]#获取该IMF的滞后天数（输入变量）
        if lag == 0:
            print("The lag of sub-signal({:.0f})".format(i)+" equals to 0")
            continue
        samples_columns=[]
        for l in range(1,lag+1):
            samples_columns.append('X'+str(l))
        samples_columns.append('Y')
        dev_test_df = pd.DataFrame()#测试集样本:输入为分解序列，输出为分解序列
        for j in range(start,stop+1):#遍历每一个附加分解结果
            data = pd.read_csv(source_path+decomposer+'_appended_test'+str(j)+'.csv')#取出逐天将test数据附加到traindev后的附加数据集分解的结果
            imf = data[columns[i-1]]#取出这一天之前相应的IMF
            nparr = np.array(imf)#转换为numpy array
            inputs = pd.DataFrame()#这一天之前分解结果形成的训练样本的输入
            for k in range(lag):#逐个生成输入特征变量
                x = pd.DataFrame(
                    nparr[k:nparr.size - (lag - k)],
                    columns=['X' + str(k + 1)])['X' + str(k + 1)]
                inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))#合并输入特征变量
            label = pd.DataFrame(nparr[lag:], columns=['Y'])['Y']#生成输出标签列
            full_data_set = pd.DataFrame(pd.concat([inputs, label], axis=1))#合并输入与输出，生成样本
            last_imf = full_data_set.iloc[full_data_set.shape[0]-1:]#取出最后一行样本作为该天的样本
            dev_test_df = pd.concat([dev_test_df,last_imf],axis=0)#合并所有天的样本
        dev_test_df=dev_test_df.reset_index(drop=True)#重新索引测试集样本
        # 用子序列训练集的归一化参数来归一化测试集
        if is_sampled:
            norm_id = pd.read_csv(save_path+'norm_sample_id_imf' + str(i) + '.csv')
        else:
            norm_id = pd.read_csv(save_path+'norm_unsample_id_imf' + str(i) + '.csv')

        series_max = (norm_id['series_max']).values
        series_min = (norm_id['series_min']).values
        dev_test_df=dev_test_df.values
        dev_test_df = 2 * (dev_test_df - series_min) / (series_max - series_min) - 1
        dev_test_df = pd.DataFrame(dev_test_df,columns=samples_columns)
        dev_samples = dev_test_df.iloc[0:dev_test_df.shape[0]-test_len]
        test_samples = dev_test_df.iloc[dev_test_df.shape[0]-test_len:]
        if header and index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(i)+'.csv') 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv') 
        elif not header and index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(i)+'.csv',header=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',header=None) 
        elif header and not index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(i)+'.csv',index=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',index=None) 
        else:
            dev_samples.to_csv(save_path+'minmax_unsample_dev_imf'+str(i)+'.csv',header=None,index=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',header=None,index=None) 

def gen_multi_step_dev2_test_samples_minmax_one_by_one(
    source_path,
    decomposer,
    lags,
    columns,
    save_path,
    start,
    stop,
    test_len,
    is_sampled=False,
    header=True,
    index=False,
):
    """ 
    Generate multi-step decomposition-ensemble testing samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -source_path: The source path of appended testing decomposition files for generate the testing samples.
        -decomposer: The toolkit used for decomposing the raw time series ('vmd','eemd','wa','ssa').
        -lags: The lags for autoregression, e.g., [1,2,3,4,5].
        -columns: the columns' name for read the source data by pandas, e.g., ['a','b','c','d','e'].
        -save_path: The path of restoring the testing samples.
        -start: The start index of appended testing decomposition.
        -stop: The stop index of appended testing decomposition.
        -test_len: The length of testing samples.
        -is_sampled: Boolean, wether or not the training samples have been sampled.
        -header: Boolean, decide wether or not save with header.
        -index: Boolean, Decide wether or not save with index.
    """
    print("Generating muliti-step forecast decomposition-ensemble\n development(2) and testing samples ...")
    for i in range(1,len(lags)+1):#imfs
        lag=lags[i-1]#获取该IMF的滞后天数（输入变量）
        if lag ==0:
            print("The lag of sub-signal({:.0f})".format(i)+" equals to 0")
            continue
        samples_columns=[]
        for l in range(1,lag+1):
            samples_columns.append('X'+str(l))
        samples_columns.append('Y')
        dev_test_df = pd.DataFrame()#测试集样本:输入为分解序列，输出为分解序列
        for j in range(start,stop+1):#遍历每一个附加分解结果
            data = pd.read_csv(source_path+decomposer+'_appended_test'+str(j)+'.csv')#取出逐天将test数据附加到traindev后的附加数据集分解的结果
            imf = data[columns[i-1]]#取出这一天之前相应的IMF
            nparr = np.array(imf)#转换为numpy array
            inputs = pd.DataFrame()#这一天之前分解结果形成的训练样本的输入
            for k in range(lag):#逐个生成输入特征变量
                x = pd.DataFrame(
                    nparr[k:nparr.size - (lag - k)],
                    columns=['X' + str(k + 1)])['X' + str(k + 1)]
                inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))#合并输入特征变量
            label = pd.DataFrame(nparr[lag:], columns=['Y'])['Y']#生成输出标签列
            full_data_set = pd.DataFrame(pd.concat([inputs, label], axis=1))#合并输入与输出，生成样本
            last_imf = full_data_set.iloc[full_data_set.shape[0]-1:]#取出最后一行样本作为该天的样本
            dev_test_df = pd.concat([dev_test_df,last_imf],axis=0)#合并所有天的样本
        dev_test_df=dev_test_df.reset_index(drop=True)#重新索引测试集样本
        # 用子序列训练集的归一化参数来归一化测试集
        if is_sampled:
            norm_id = pd.read_csv(save_path+'norm_sample_id_imf' + str(i) + '.csv')
        else:
            norm_id = pd.read_csv(save_path+'norm_unsample_id_imf' + str(i) + '.csv')
        series_max = (norm_id['series_max']).values
        series_min = (norm_id['series_min']).values
        dev_test_df=dev_test_df.values
        dev_test_df = 2 * (dev_test_df - series_min) / (series_max - series_min) - 1
        dev_test_df = pd.DataFrame(dev_test_df,columns=samples_columns)
        dev_samples = dev_test_df.iloc[0:dev_test_df.shape[0]-test_len]
        test_samples = dev_test_df.iloc[dev_test_df.shape[0]-test_len:]
        if header and index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev2_imf'+str(i)+'.csv') 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv') 
        elif not header and index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev2_imf'+str(i)+'.csv',header=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',header=None) 
        elif header and not index:
            dev_samples.to_csv(save_path+'minmax_unsample_dev2_imf'+str(i)+'.csv',index=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',index=None) 
        else:
            dev_samples.to_csv(save_path+'minmax_unsample_dev2_imf'+str(i)+'.csv',header=None,index=None) 
            test_samples.to_csv(save_path+'minmax_unsample_test_imf'+str(i)+'.csv',header=None,index=None) 

# @deprecated
# def pca_one_step(
#     station,
#     decomposer,
#     predict_pattern,
#     n_components,
#     header=True,
#     index=False,
# ):
#     """
#     Reduce input dimension using Principal Component Analysis.
#     \nArgs:
#     \n-station: String, the station where the orignal time series come from.
#     \n-decomposer: String, the decomposer used for decomposing the original time series.
#     \n-predict_pattern: String, the prediction pattern (hindcast or forecast) that PCA work on.
#     \n-n_components: Number of components to keep. If n_components is not set, all the 
#     components are kept. If n_components=='mle', Minka's MLE is used to guess the dimension.
#     If 0 < n_compoents <1 and svd_solver == 'full', select the components such that
#     the mount of variance that need to be explained is greater thhan the percentage specified by n_components.
#     \n-header: Boolean, decide wether or not save with header.
#     \n-index: Boolean, Decide wether or not save with index.
#     """
#     signals = station+'_'+decomposer
#     # load one-step one-month forecast or hindcast samples and the normalization indicators
#     train = pd.read_csv(root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'/minmax_unsample_train.csv')
#     dev = pd.read_csv(root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'/minmax_unsample_dev.csv')
#     test = pd.read_csv(root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'/minmax_unsample_test.csv')
#     norm_id = pd.read_csv(root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'/norm_unsample_id.csv')
#     sMax = (norm_id['series_max']).values
#     sMin = (norm_id['series_min']).values
#     # Conncat the training, development and testing samples
#     samples = pd.concat([train,dev,test],axis=0)
#     samples = samples.reset_index(drop=True)
#     # Renormalized the entire samples
#     samples = np.multiply(samples + 1,sMax - sMin) / 2 + sMin

#     y = samples['Y']
#     X = samples.drop('Y',axis=1)
#     print("Input features before PAC:\n{}".format(X.tail()))
#     pca = decomposition.PCA(n_components=n_components)
#     pca.fit(X)
#     pca_X = pca.transform(X)
#     columns=[]
#     for i in range(1,pca_X.shape[1]+1):
#         columns.append('X'+str(i))
#     pca_X = pd.DataFrame(pca_X,columns=columns)
#     print("Input features after PAC:\n{}".format(pca_X.tail()))

#     pca_samples = pd.concat([pca_X,y],axis=1)
#     pca_train = pca_samples.iloc[:train.shape[0]]
#     pca_train=pca_train.reset_index(drop=True)
#     pca_dev = pca_samples.iloc[train.shape[0]:train.shape[0]+dev.shape[0]]
#     pca_dev=pca_dev.reset_index(drop=True)
#     pca_test = pca_samples.iloc[train.shape[0]+dev.shape[0]:]
#     pca_test=pca_test.reset_index(drop=True)

#     series_min = pca_train.min(axis=0)
#     series_max = pca_train.max(axis=0)
#     pca_train = 2 * (pca_train - series_min) / (series_max - series_min) - 1
#     pca_dev = 2 * (pca_dev - series_min) / (series_max - series_min) - 1
#     pca_test = 2 * (pca_test - series_min) / (series_max - series_min) - 1


#     # Generate pandas series for series' mean and standard devation
#     series_max = pd.DataFrame(series_max, columns=['series_max'])
#     series_min = pd.DataFrame(series_min, columns=['series_min'])
#     # Merge max serie and min serie
#     normalize_indicators = pd.concat([series_max, series_min], axis=1)
#     if isinstance(n_components,str):
#         pca_data_path = root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'_with_pca_'+n_components+'/'
#     else:
#         pca_data_path = root_path+'/'+signals+'/data/one_step_one_month_'+predict_pattern+'_with_pca_'+str(n_components)+'/'

#     if not os.path.exists(pca_data_path):
#         os.makedirs(pca_data_path)
#     normalize_indicators.to_csv(pca_data_path+'norm_unsample_id.csv')
#     if header and index:
#         pca_train.to_csv(pca_data_path+'minmax_unsample_train.csv')
#         pca_dev.to_csv(pca_data_path+'minmax_unsample_dev.csv')
#         pca_test.to_csv(pca_data_path+'minmax_unsample_test.csv')
#     elif header and not index:
#         pca_train.to_csv(pca_data_path+'minmax_unsample_train.csv',index=None)
#         pca_dev.to_csv(pca_data_path+'minmax_unsample_dev.csv',index=None)
#         pca_test.to_csv(pca_data_path+'minmax_unsample_test.csv',index=None)
#     elif not header and index:
#         pca_train.to_csv(pca_data_path+'minmax_unsample_train.csv',header=None)
#         pca_dev.to_csv(pca_data_path+'minmax_unsample_dev.csv',header=None)
#         pca_test.to_csv(pca_data_path+'minmax_unsample_test.csv',header=None)
#     elif not header and not index:
#         pca_train.to_csv(pca_data_path+'minmax_unsample_train.csv',header=None,index=None)
#         pca_dev.to_csv(pca_data_path+'minmax_unsample_dev.csv',header=None,index=None)
#         pca_test.to_csv(pca_data_path+'minmax_unsample_test.csv',header=None,index=None)


def gen_one_step_long_leading_samples_minmax(
    station,
    decomposer,
    leading_time,
    input_columns,
    start,
    stop,
    test_len,
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
    save_path = data_path+"one_step_"+str(leading_time)+"_month_forecast/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lag=20+leading_time
    train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
    train_decompositions= pd.read_csv(train_decompose_file)
    orig = train_decompositions[output_clolumn][lag:]
    orig=orig.reset_index(drop=True)
    selected = {}
    input_df = pd.DataFrame()
    for col in input_columns:
        subsignal = np.array(train_decompositions[col])
        inputs = pd.DataFrame()
        for k in range(lag):
            x = pd.DataFrame(subsignal[k:subsignal.size-(lag-k)],columns=["X"+str(k+1)])["X"+str(k+1)]
            x=x.reset_index(drop=True)
            inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))
        # label = pd.DataFrame(subsignal[lag:], columns=['Y'])['Y']
        # label=label.reset_index(drop=True)
        partin_out = pd.concat([inputs,orig],axis=1)
        corrs=partin_out.corr(method="pearson")
        corrs = corrs.iloc[0:20]
        # print(corrs)
        bools = (corrs["ORIG"]>=filter_boundary) | (corrs["ORIG"]<=-filter_boundary)
        select=list((corrs.loc[bools==True]).index.values)
        selected[col]=select
        input_df = pd.concat([input_df,inputs[select]],axis=1)
    columns = []
    for i in range(0,input_df.shape[1]):
        columns.append("X"+str(i+1))
    columns.append("Y")

    train_samples = pd.DataFrame((pd.concat([input_df,orig],axis=1)).values,columns=columns)
    # print(input_df.tail())
    # print(selected)
    # print(train_samples.tail())
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
    print(dev_test_samples)
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
    
    gen_one_step_long_leading_samples_minmax(
        station="Huaxian",
        decomposer="eemd",
        leading_time=7,
        input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9',],
        start = 553,
        stop= 792,#792
        test_len=120,
        filter_boundary=0.2,
    )

    # gen_one_step_long_leading_samples_minmax(
    #     station="Huaxian",
    #     decomposer="vmd",
    #     leading_time=9,
    #     filter_boundary=0.2,
    #     input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',]
    # )

    # gen_one_step_long_leading_samples_minmax(
    #     station="Huaxian",
    #     decomposer="ssa",
    #     leading_time=9,
    #     filter_boundary=0.2,
    #     input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise']
    # )

    # gen_one_step_long_leading_samples_minmax(
    #     station="Huaxian",
    #     decomposer="wd",
    #     leading_time=9,
    #     filter_boundary=0.2,
    #     input_columns=['D1','D2','A2',]
    # )