import os
root_path = os.path.dirname(os.path.abspath('__file__'))

import sys
import glob
import pandas as pd
import numpy as np
from sklearn import decomposition
import deprecated
import logging
sys.path.append(root_path)
from config.globalLog import logger


def generate_monoscale_samples(source_file, save_path, lags_dict, column, test_len, lead_time=1,regen=False):
    """Generate learning samples for autoregression problem using original time series. 
    Args:
    'source_file' -- ['String'] The source data file path.
    'save_path' --['String'] The path to restore the training, development and testing samples.
    'lags_dict' -- ['int dict'] The lagged time for original time series.
    'column' -- ['String']The column's name for read the source data by pandas.
    'test_len' --['int'] The length of development and testing set.
    'lead_time' --['int'] The lead time.
    """
    logger.info('Generating muliti-step decomposition-ensemble hindcasting samples')
    save_path = save_path+'/'+str(lead_time)+'_ahead_pacf/'
    logger.info('Source file:{}'.format(source_file))
    logger.info('Save path:{}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        #  Load data from local dick
        if '.xlsx' in source_file:
            dataframe = pd.read_excel(source_file)[column]
        elif '.csv' in source_file:
            dataframe = pd.read_csv(source_file)[column]
        # convert pandas dataframe to numpy array
        nparr = np.array(dataframe)
        # Create an empty pandas Dataframe
        full_samples = pd.DataFrame()
        # Generate input series based on lag and add these series to full dataset
        lag = lags_dict['ORIG']
        for i in range(lag):
            x = pd.DataFrame(nparr[i:dataframe.shape[0] -
                                   (lag - i)], columns=['X' + str(i + 1)])
            x = x.reset_index(drop=True)
            full_samples = pd.concat([full_samples, x], axis=1, sort=False)

        # Generate label data
        label = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])
        label = label.reset_index(drop=True)
        full_samples = full_samples[:full_samples.shape[0]-(lead_time-1)]
        full_samples = full_samples.reset_index(drop=True)
        # Add labled data to full_data_set
        full_samples = pd.concat([full_samples, label], axis=1, sort=False)
        # Get the length of this series
        series_len = full_samples.shape[0]
        # Get the training and developing set
        train_dev_samples = full_samples[0:(series_len - test_len)]
        # Get the testing set.
        test_samples = full_samples[(series_len - test_len):series_len]
        # train_dev_len = train_dev_samples.shape[0]
        train_samples = full_samples[0:(series_len - test_len - test_len)]
        dev_samples = full_samples[(
            series_len - test_len - test_len):(series_len - test_len)]
        assert (train_samples.shape[0] + dev_samples.shape[0] +
                test_samples.shape[0]) == series_len
        # Get the max and min value of each series
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_samples = 2 * (train_samples - series_min) / \
            (series_max - series_min) - 1
        dev_samples = 2 * (dev_samples - series_min) / \
            (series_max - series_min) - 1
        test_samples = 2 * (test_samples - series_min) / \
            (series_max - series_min) - 1

        logger.info('Series length:{}'.format(series_len))
        logger.info('Series length:{}'.format(series_len))
        logger.info(
            'Training-development sample size:{}'.format(train_dev_samples.shape[0]))
        logger.info('Training sample size:{}'.format(train_samples.shape[0]))
        logger.info('Development sample size:{}'.format(dev_samples.shape[0]))
        logger.info('Testing sample size:{}'.format(test_samples.shape[0]))

        series_max = pd.DataFrame(series_max, columns=['series_max'])
        series_min = pd.DataFrame(series_min, columns=['series_min'])
        normalize_indicators = pd.concat([series_max, series_min], axis=1)
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)


def gen_one_step_hindcast_samples(station, decomposer, lags_dict, input_columns, output_column, test_len,
                                  wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate one step hindcast decomposition-ensemble learning samples. 
    Args:
    'station'-- ['string'] The station where the original time series come from.
    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.
    'lags_dict'-- ['int dict'] The lagged time for each subsignal.
    'input_columns'-- ['string list'] The input columns' name used for generating the learning samples.
    'output_columns'-- ['string'] The output column's name used for generating the learning samples.
    'test_len'-- ['int'] The size of development and testing samples ().
    """
    logger.info('Generating one-step decomposition ensemble hindcasting samples')
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Input columns:{}'.format(input_columns))
    logger.info('Output column:{}'.format(output_column))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    #  Load data from local dick
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_"+str(lead_time)+"_ahead_hindcast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
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
        max_lag = max(lags_dict.values())
        samples_size = data_size-max_lag
        # Generate feature columns
        samples_cols = []
        for i in range(sum(lags_dict.values())):
            samples_cols.append('X'+str(i+1))
        samples_cols.append('Y')
        # Generate input colmuns for each subsignal
        full_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one subsignal
            one_in = (input_data[input_columns[i]]).values
            oness = pd.DataFrame()
            lag = lags_dict[input_columns[i]]
            for j in range(lag):
                x = pd.DataFrame(one_in[j:data_size-(lag-j)],
                                 columns=['X' + str(j + 1)])
                x = x.reset_index(drop=True)
                oness = pd.concat([oness, x], axis=1, sort=False)
            # make all sample size of each subsignal identical
            oness = oness.iloc[oness.shape[0]-samples_size:]
            oness = oness.reset_index(drop=True)
            full_samples = pd.concat([full_samples, oness], axis=1, sort=False)
        # Get the target
        target = (output_data.values)[max_lag+lead_time-1:]
        target = pd.DataFrame(target, columns=['Y'])
        full_samples = full_samples[:full_samples.shape[0]-(lead_time-1)]
        full_samples = full_samples.reset_index(drop=True)
        # Concat the features and target
        full_samples = pd.concat([full_samples, target], axis=1, sort=False)
        full_samples = pd.DataFrame(full_samples.values, columns=samples_cols)
        full_samples.to_csv(save_path+'full_samples.csv')
        assert samples_size == full_samples.shape[0]
        # Get the training and developing set
        train_dev_samples = full_samples[0:(samples_size - test_len)]
        # Get the testing set.
        test_samples = full_samples[(samples_size - test_len):samples_size]
        # train_dev_len = train_dev_samples.shape[0]
        train_samples = full_samples[0:(samples_size - test_len - test_len)]
        dev_samples = full_samples[(
            samples_size - test_len - test_len):(samples_size - test_len)]
        assert (train_samples['X1'].size + dev_samples['X1'].size +
                test_samples['X1'].size) == samples_size
        # Get the max and min value of training set
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_samples = 2 * (train_samples - series_min) / \
            (series_max - series_min) - 1
        dev_samples = 2 * (dev_samples - series_min) / \
            (series_max - series_min) - 1
        test_samples = 2 * (test_samples - series_min) / \
            (series_max - series_min) - 1

        logger.info('Save path:{}'.format(save_path))
        logger.info('Series length:{}'.format(samples_size))
        logger.info('Training and development sample size:{}'.format(
            train_dev_samples.shape[0]))
        logger.info('Training sample size:{}'.format(train_samples.shape[0]))
        logger.info('Development sample size:{}'.format(dev_samples.shape[0]))
        logger.info('Testing sample size:{}'.format(test_samples.shape[0]))


        series_max = pd.DataFrame(series_max, columns=['series_max'])
        series_min = pd.DataFrame(series_min, columns=['series_min'])
        normalize_indicators = pd.concat([series_max, series_min], axis=1)
        normalize_indicators.to_csv(save_path+'norm_unsample_id.csv')
        train_samples.to_csv(save_path + 'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path + 'minmax_unsample_dev.csv', index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)


def gen_one_step_forecast_samples_triandev_test(station, decomposer, lags_dict, input_columns, output_column, start, stop, test_len,
                                                wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate one step forecast decomposition-ensemble samples. 
    Args:
    'station'-- ['string'] The station where the original time series come from.
    'decomposer'-- ['string'] The decompositin algorithm used for decomposing the original time series.
    'lags_dict'-- ['int dict'] The lagged time for subsignals.
    'input_columns'-- ['string lsit'] the input columns' name for read the source data by pandas.
    'output_columns'-- ['string'] the output column's name for read the source data by pandas.
    'start'-- ['int'] The start index of appended decomposition file.
    'stop'-- ['int'] The stop index of appended decomposotion file.
    'test_len'-- ['int'] The size of development and testing samples.
    """
    logger.info(
        'Generateing one-step decomposition ensemble forecasting samples (traindev-test pattern)')
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Input columns:{}'.format(input_columns))
    logger.info('Output column:{}'.format(output_column))
    logger.info('Validation start index:{}'.format(start))
    logger.info('Validation stop index:{}'.format(stop))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    #  Load data from local dick
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"one_step_" + \
        str(lead_time)+"_ahead_forecast_pacf_traindev_test/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
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
        max_lag = max(lags_dict.values())
        traindev_samples_size = traindev_data_size-max_lag
        # Generate feature columns
        samples_cols = []
        for i in range(sum(lags_dict.values())):
            samples_cols.append('X'+str(i+1))
        samples_cols.append('Y')
        # Generate input colmuns for each input feature
        train_dev_samples = pd.DataFrame()
        for i in range(subsignals_num):
            # Get one input feature
            one_in = (traindev_input_data[input_columns[i]]).values  # subsignal
            lag = lags_dict[input_columns[i]]
            oness = pd.DataFrame()  # restor input features
            for j in range(lag):
                x = pd.DataFrame(one_in[j:traindev_data_size-(lag-j)],
                                 columns=['X' + str(j + 1)])['X' + str(j + 1)]
                x = x.reset_index(drop=True)
                oness = pd.DataFrame(pd.concat([oness, x], axis=1))
            oness = oness.iloc[oness.shape[0]-traindev_samples_size:]
            oness = oness.reset_index(drop=True)
            train_dev_samples = pd.DataFrame(
                pd.concat([train_dev_samples, oness], axis=1))
        # Get the target
        target = (traindev_output_data.values)[max_lag+lead_time-1:]
        target = pd.DataFrame(target, columns=['Y'])
        train_dev_samples = train_dev_samples[:traindev_samples_size-(lead_time-1)]
        train_dev_samples = train_dev_samples.reset_index(drop=True)
        # Concat the features and target
        train_dev_samples = pd.concat([train_dev_samples, target], axis=1)
        train_dev_samples = pd.DataFrame(
            train_dev_samples.values, columns=samples_cols)
        train_dev_samples.to_csv(save_path+'train_dev_samples.csv')
        train_samples = train_dev_samples[:train_dev_samples.shape[0]-120]
        dev_samples = train_dev_samples[train_dev_samples.shape[0]-120:]
        assert traindev_samples_size == train_dev_samples.shape[0]
        # normalize the train_samples
        series_max = train_samples.max(axis=0)
        series_min = train_samples.min(axis=0)
        # Normalize each series to the range between -1 and 1
        train_samples = 2 * (train_samples - series_min) / \
            (series_max - series_min) - 1
        dev_samples = 2 * (dev_samples - series_min) / \
            (series_max - series_min) - 1
        test_samples = pd.DataFrame()
        appended_file_path = data_path+decomposer+"-test/"
        for k in range(start, stop+1):
            #  Load data from local dick
            appended_decompositions = pd.read_csv(
                appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
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
            samples_size = data_size-max_lag
            # Generate input colmuns for each subsignal
            appended_samples = pd.DataFrame()
            for i in range(subsignals_num):
                # Get one subsignal
                one_in = (input_data[input_columns[i]]).values
                lag = lags_dict[input_columns[i]]
                oness = pd.DataFrame()
                for j in range(lag):
                    x = pd.DataFrame(
                        one_in[j:data_size-(lag-j)], columns=['X' + str(j + 1)])['X' + str(j + 1)]
                    x = x.reset_index(drop=True)
                    oness = pd.DataFrame(pd.concat([oness, x], axis=1))
                oness = oness.iloc[oness.shape[0]-samples_size:]
                oness = oness.reset_index(drop=True)
                appended_samples = pd.DataFrame(
                    pd.concat([appended_samples, oness], axis=1))
            # Get the target
            target = (output_data.values)[max_lag+lead_time-1:]
            target = pd.DataFrame(target, columns=['Y'])
            appended_samples = appended_samples[:
                                                appended_samples.shape[0]-(lead_time-1)]
            appended_samples = appended_samples.reset_index(drop=True)
            # Concat the features and target
            appended_samples = pd.concat([appended_samples, target], axis=1)
            appended_samples = pd.DataFrame(
                appended_samples.values, columns=samples_cols)
            # Get the last sample of full samples
            last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
            test_samples = pd.concat([test_samples, last_sample], axis=0)
        test_samples = test_samples.reset_index(drop=True)
        test_samples.to_csv(save_path+'test_samples.csv')
        test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1
        assert test_len == test_samples.shape[0]
        logger.info('Save path:{}'.format(save_path))
        logger.info('The size of training samples:{}'.format(
            train_samples.shape[0]))
        logger.info('The size of development samples:{}'.format(
            dev_samples.shape[0]))
        logger.info('The size of testing samples:{}'.format(test_samples.shape[0]))


        series_max = pd.DataFrame(series_max, columns=['series_max'])
        series_min = pd.DataFrame(series_min, columns=['series_min'])
        normalize_indicators = pd.concat([series_max, series_min], axis=1)
        normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
        train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
        dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
        test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)


def gen_one_step_forecast_samples(station, decomposer, lags_dict, input_columns, output_column, start, stop, test_len,
                                  wavelet_level="db10-2", lead_time=1, mode='PACF', pre_times=20, filter_boundary=0.2, n_components=None,regen=False):
    """ 
    Generate one step forecast decomposition-ensemble samples based on 
    Partial autocorrelation function (PACF), Pearson coefficient correlation (pearson).
    Set n_components to 'mle' or an integer to perform principle component analysis (PCA).
    Args:
    'station'-- ['string'] The station where the original time series come from.
    'decomposer'-- ['string'] The decomposition algorithm used for decomposing the original time series.
    'lags_dict'-- ['int dict'] The lagged time for subsignals in 'PACF' mode.
    'input_columns'-- ['string list'] the input columns' name for read the source data by pandas.
    'output_column'-- ['string'] the output column's name for read the source data by pandas.
    'start'-- ['int'] The start index of appended decomposition file.
    'stop'-- ['int'] The stop index of appended decomposition file.
    'test_len'-- ['int'] The size of development and testing samples.
    'wavelet_level'-- ['String'] The mother wavelet and decomposition level of DWT.
    'lead_time'-- ['int'] The lead time for auto regression models.
    'mode'-- ['String'] The samples generation mode, i.e., "PACF" and "Pearson", for auto regression models.
    'pre_times'-- ['int'] The lag times for compute Pearson coefficient correlation.
    'filter_boundary'-- ['float'] The filter threshold of Pearson coefficient correlation for selecting input predictors.
    'n_components'-- ['String or int'] The number of reserved components in PCA. If n_components is set to None, PCA will not be performed.
    """
    logger.info(
        "Generateing one-step decomposition ensemble forecasting samples (train-devtest pattern)")
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Input columns:{}'.format(input_columns))
    logger.info('Output column:{}'.format(output_column))
    logger.info('Validation start index:{}'.format(start))
    logger.info('Validation stop index:{}'.format(stop))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    logger.info('Generation mode:{}'.format(mode))
    logger.info('Selected previous lag times:{}'.format(pre_times))
    logger.info(
        'Filter threshold of predictors selection:{}'.format(filter_boundary))
    logger.info('Number of components for PCA:{}'.format(n_components))
    #  Load data from local dick
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    if mode == 'PACF' and n_components == None:
        save_path = data_path+"one_step_" + \
            str(lead_time)+"_ahead_forecast_pacf/"
    elif mode == 'PACF' and n_components != None:
        save_path = data_path+"one_step_" + \
            str(lead_time)+"_ahead_forecast_pacf_pca"+str(n_components)+"/"
    elif mode == 'Pearson' and n_components == None:
        save_path = data_path+"one_step_" + \
            str(lead_time)+"_ahead_forecast_pearson"+str(filter_boundary)+"/"
    elif mode == 'Pearson' and n_components != None:
        save_path = data_path+"one_step_" + \
            str(lead_time)+"_ahead_forecast_pearson" + \
            str(filter_boundary)+"_pca"+str(n_components)+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        # !!!!!!Generate training samples
        if mode == 'PACF':
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

            max_lag = max(lags_dict.values())
            logger.debug('max lag:{}'.format(max_lag))
            train_samples_size = train_data_size-max_lag
            # Generate feature columns
            samples_cols = []
            for i in range(sum(lags_dict.values())):
                samples_cols.append('X'+str(i+1))
            samples_cols.append('Y')
            # Generate input colmuns for each input feature
            train_samples = pd.DataFrame()
            for i in range(subsignals_num):
                # Get one input feature
                one_in = (train_input_data[input_columns[i]]).values  # subsignal
                lag = lags_dict[input_columns[i]]
                logger.debug('lag:{}'.format(lag))
                oness = pd.DataFrame()  # restor input features
                for j in range(lag):
                    x = pd.DataFrame(one_in[j:train_data_size-(lag-j)], columns=['X' + str(j + 1)])
                    x = x.reset_index(drop=True)
                    oness = pd.concat([oness, x], axis=1, sort=False)
                logger.debug("oness:\n{}".format(oness))
                oness = oness.iloc[oness.shape[0]-train_samples_size:]
                oness = oness.reset_index(drop=True)
                train_samples = pd.concat([train_samples, oness], axis=1, sort=False)
            # Get the target
            target = (train_output_data.values)[max_lag+lead_time-1:]
            target = pd.DataFrame(target, columns=['Y'])
            # Concat the features and target
            train_samples = train_samples[:train_samples.shape[0]-(lead_time-1)]
            train_samples = train_samples.reset_index(drop=True)
            train_samples = pd.concat([train_samples, target], axis=1)
            train_samples = pd.DataFrame(train_samples.values, columns=samples_cols)
            train_samples.to_csv(save_path+'train_samples.csv')
            # assert train_samples_size == train_samples.shape[0]
            # !!!!!!!!!!!Generate development and testing samples
            dev_test_samples = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for k in range(start, stop+1):
                #  Load data from local dick
                appended_decompositions = pd.read_csv(
                    appended_file_path+decomposer+'_appended_test'+str(k)+'.csv')
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
                samples_size = data_size-max_lag
                # Generate input colmuns for each subsignal
                appended_samples = pd.DataFrame()
                for i in range(subsignals_num):
                    # Get one subsignal
                    one_in = (input_data[input_columns[i]]).values
                    lag = lags_dict[input_columns[i]]
                    oness = pd.DataFrame()
                    for j in range(lag):
                        x = pd.DataFrame(
                            one_in[j:data_size-(lag-j)], columns=['X' + str(j + 1)])
                        x = x.reset_index(drop=True)
                        oness = pd.concat([oness, x], axis=1, sort=False)
                    oness = oness.iloc[oness.shape[0]-samples_size:]
                    oness = oness.reset_index(drop=True)
                    appended_samples = pd.concat(
                        [appended_samples, oness], axis=1, sort=False)
                # Get the target
                target = (output_data.values)[max_lag+lead_time-1:]
                target = pd.DataFrame(target, columns=['Y'])
                # Concat the features and target
                appended_samples = appended_samples[:
                                                    appended_samples.shape[0]-(lead_time-1)]
                appended_samples = appended_samples.reset_index(drop=True)
                appended_samples = pd.concat(
                    [appended_samples, target], axis=1, sort=False)
                appended_samples = pd.DataFrame(
                    appended_samples.values, columns=samples_cols)
                # Get the last sample of full samples
                last_sample = appended_samples.iloc[appended_samples.shape[0]-1:]
                dev_test_samples = pd.concat(
                    [dev_test_samples, last_sample], axis=0)
            dev_test_samples = dev_test_samples.reset_index(drop=True)
            dev_test_samples.to_csv(save_path+'dev_test_samples.csv')
            dev_samples = dev_test_samples.iloc[0: dev_test_samples.shape[0]-test_len]
            test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]

            if n_components != None:
                logger.info('Performa PCA on samples based on PACF')
                samples = pd.concat([train_samples, dev_samples, test_samples], axis=0, sort=False)
                samples = samples.reset_index(drop=True)
                y = samples['Y']
                X = samples.drop('Y', axis=1)
                logger.debug('X contains Nan:{}'.format(X.isnull().values.any()))
                logger.debug("Input features before PAC:\n{}".format(X))
                pca = decomposition.PCA(n_components=n_components)
                pca.fit(X)
                pca_X = pca.transform(X)
                columns = []
                for i in range(1, pca_X.shape[1]+1):
                    columns.append('X'+str(i))
                pca_X = pd.DataFrame(pca_X, columns=columns)
                logger.debug("Input features after PAC:\n{}".format(pca_X.tail()))
                pca_samples = pd.concat([pca_X, y], axis=1)
                train_samples = pca_samples.iloc[:train_samples.shape[0]]
                train_samples = train_samples.reset_index(drop=True)
                logger.debug('Training samples after PCA:\n{}'.format(train_samples))
                dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                dev_samples = dev_samples.reset_index(drop=True)
                logger.debug('Development samples after PCA:\n{}'.format(dev_samples))
                test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                test_samples = test_samples.reset_index(drop=True)
                logger.debug('Testing samples after PCA:\n{}'.format(test_samples))

            # Normalize each series to the range between -1 and 1
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            train_samples = 2 * (train_samples - series_min) / \
                (series_max - series_min) - 1
            dev_samples = 2 * (dev_samples-series_min) / \
                (series_max-series_min) - 1
            test_samples = 2 * (test_samples-series_min) / \
                (series_max-series_min) - 1

            logger.info('Save path:{}'.format(save_path))
            logger.info('The size of training samples:{}'.format(
                train_samples.shape[0]))
            logger.info('The size of development samples:{}'.format(
                dev_samples.shape[0]))
            logger.info('The size of testing samples:{}'.format(
                test_samples.shape[0]))

            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
            train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
            test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)

        elif mode == 'Pearson':
            # lag pre_times+lead_time(e.g.,30+3)
            lag = pre_times+lead_time
            pre_cols = []
            for i in range(1, pre_times+1):
                pre_cols.append("X"+str(i))
            logger.debug("Previous columns of lagged months:\n{}".format(pre_cols))
            train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
            train_decompositions = pd.read_csv(train_decompose_file)
            orig = train_decompositions[output_column][lag:]
            orig = orig.reset_index(drop=True)
            selected = {}
            input_df = pd.DataFrame()
            for col in input_columns:
                logger.debug("Perform subseries:{}".format(col))
                subsignal = np.array(train_decompositions[col])
                inputs = pd.DataFrame()
                for k in range(lag):
                    x = pd.DataFrame(
                        subsignal[k:subsignal.size-(lag-k)], columns=["X"+str(k+1)])["X"+str(k+1)]
                    x = x.reset_index(drop=True)
                    inputs = pd.DataFrame(pd.concat([inputs, x], axis=1))
                pre_inputs = inputs[pre_cols]
                logger.debug("Previous inputs:\n{}".format(pre_inputs.head()))
                partin_out = pd.concat([pre_inputs, orig], axis=1)
                logger.debug(
                    "Partial inputs and output:\n{}".format(partin_out.head()))
                corrs = partin_out.corr(method="pearson")
                logger.debug("Entire pearson coefficients:\n{}".format(corrs))
                corrs = (corrs[output_column]).iloc[0:corrs.shape[0]-1]
                orig_corrs = corrs.squeeze()
                logger.debug("Selected pearson coefficients:\n{}".format(orig_corrs))
                bools = abs(orig_corrs) >= filter_boundary
                logger.debug("Conditions judge:{}".format(bools))
                select = list((orig_corrs.loc[bools == True]).index.values)
                logger.debug("Selected inputs:\n{}".format(select))
                selected[col] = select
                input_df = pd.concat([input_df, pre_inputs[select]], axis=1)
            logger.debug("Selected inputs:\n{}".format(selected))
            logger.debug("Entire inputs:\n{}".format(input_df.head()))
            columns = []
            for i in range(0, input_df.shape[1]):
                columns.append("X"+str(i+1))
            columns.append("Y")

            train_samples = pd.DataFrame(
                (pd.concat([input_df, orig], axis=1)).values, columns=columns)
            dev_test_samples = pd.DataFrame()
            for i in range(start, stop+1):
                append_decompositions = pd.read_csv(
                    data_path+decomposer+"-test/"+decomposer+"_appended_test"+str(i)+".csv")
                append_orig = append_decompositions[output_column][lag:]
                append_orig = append_orig.reset_index(drop=True)
                append_input_df = pd.DataFrame()
                for col in input_columns:
                    append_subsignal = np.array(append_decompositions[col])
                    append_inputs = pd.DataFrame()
                    for k in range(lag):
                        x = pd.DataFrame(
                            append_subsignal[k:append_subsignal.size-(lag-k)], columns=["X"+str(k+1)])["X"+str(k+1)]
                        x = x.reset_index(drop=True)
                        append_inputs = pd.concat([append_inputs, x], axis=1)
                    append_input_df = pd.concat(
                        [append_input_df, append_inputs[selected[col]]], axis=1)
                append_samples = pd.concat([append_input_df, append_orig], axis=1)
                append_samples = pd.DataFrame(append_samples.values, columns=columns)

                last_sample = append_samples.iloc[append_samples.shape[0]-1:]
                dev_test_samples = pd.concat(
                    [dev_test_samples, last_sample], axis=0)
            dev_test_samples = dev_test_samples.reset_index(drop=True)
            dev_samples = dev_test_samples.iloc[0:
                                                dev_test_samples.shape[0]-test_len]
            test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
            dev_samples = dev_samples.reset_index(drop=True)
            test_samples = test_samples.reset_index(drop=True)

            # Perform PCA on samples based on Pearson
            if n_components != None:
                logger.info('Performa PCA on samples based on PACF')
                samples = pd.concat(
                    [train_samples, dev_samples, test_samples], axis=0, sort=False)
                samples = samples.reset_index(drop=True)
                y = samples['Y']
                X = samples.drop('Y', axis=1)
                logger.debug("Input features before PAC:\n{}".format(X.tail()))
                pca = decomposition.PCA(n_components=n_components)
                pca.fit(X)
                pca_X = pca.transform(X)
                columns = []
                for i in range(1, pca_X.shape[1]+1):
                    columns.append('X'+str(i))
                pca_X = pd.DataFrame(pca_X, columns=columns)
                logger.debug("Input features after PAC:\n{}".format(pca_X.tail()))
                pca_samples = pd.concat([pca_X, y], axis=1)
                train_samples = pca_samples.iloc[:train_samples.shape[0]]
                train_samples = train_samples.reset_index(drop=True)
                dev_samples = pca_samples.iloc[train_samples.shape[0]:train_samples.shape[0]+dev_samples.shape[0]]
                dev_samples = dev_samples.reset_index(drop=True)
                test_samples = pca_samples.iloc[train_samples.shape[0] +dev_samples.shape[0]:]
                test_samples = test_samples.reset_index(drop=True)

            # Normalize the samples
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            train_samples = 2 * (train_samples - series_min) / \
                (series_max - series_min) - 1
            dev_samples = 2*(dev_samples-series_min)/(series_max-series_min)-1
            test_samples = 2*(test_samples-series_min)/(series_max-series_min)-1

            # Save results
            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(save_path+"norm_unsample_id.csv")
            train_samples.to_csv(save_path+'minmax_unsample_train.csv', index=None)
            dev_samples.to_csv(save_path+'minmax_unsample_dev.csv', index=None)
            test_samples.to_csv(save_path+'minmax_unsample_test.csv', index=None)


def gen_multi_step_hindcast_samples(station, decomposer, lags_dict, columns, test_len,
                                    wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate muliti-step learning samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags_dict: The lags for autoregression.
        -columns: the columns' name for read the source data by pandas.
        -save_path: The path to restore the training, development and testing samples.
        -test_len: The length of validation(development or testing) set.
    """
    logger.info(
        "Generating muliti-step decompositionensemble hindcasting samples")
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Signals:{}'.format(columns))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_"+str(lead_time)+"_ahead_hindcast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        decompose_file = data_path+decomposer.upper()+"_FULL.csv"
        decompositions = pd.read_csv(decompose_file)

        for k in range(len(columns)):
            lag = lags_dict[columns[k]]
            if lag == 0:
                logger.info("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
                continue
            # Obtain decomposed sub-signal
            sub_signal = decompositions[columns[k]]
            # convert pandas dataframe to numpy array
            nparr = np.array(sub_signal)
            # Create an empty pandas Dataframe
            full_samples = pd.DataFrame()
            # Generate input series based on lag and add these series to full dataset
            for i in range(lag):
                x = pd.DataFrame(
                    nparr[i:sub_signal.shape[0] - (lag - i)], columns=['X' + str(i + 1)])
                x = x.reset_index(drop=True)
                full_samples = pd.DataFrame(pd.concat([full_samples, x], axis=1))

            # Generate label data
            label = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])['Y']
            label = label.reset_index(drop=True)
            full_samples = full_samples[:full_samples.shape[0]-(lead_time-1)]
            full_samples = full_samples.reset_index(drop=True)
            # Add labled data to full_data_set
            full_samples = pd.concat([full_samples, label], axis=1, sort=False)
            # Get the length of this series
            series_len = full_samples.shape[0]
            # Get the training and developing set
            train_dev_samples = full_samples[0:(series_len - test_len)]
            # Get the testing set.
            test_samples = full_samples[(series_len - test_len):series_len]
            # Do sampling if 'sampling' is True
            train_samples = full_samples[0:(series_len - test_len - test_len)]
            dev_samples = full_samples[(
                series_len - test_len - test_len):(series_len - test_len)]
            assert (train_samples.shape[0] + dev_samples.shape[0] +
                    test_samples.shape[0]) == series_len
            # Get the max and min value of each series
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            # Normalize each series to the range between -1 and 1
            train_samples = 2 * (train_samples - series_min) / \
                (series_max - series_min) - 1
            dev_samples = 2 * (dev_samples - series_min) / \
                (series_max - series_min) - 1
            test_samples = 2 * (test_samples - series_min) / \
                (series_max - series_min) - 1

            logger.info('Series length:{}'.format(series_len))
            logger.info('Save path:{}'.format(save_path))
            logger.info('The size of training and development samples:{}'.format(
                train_dev_samples.shape[0]))
            logger.info('The size of training samples:{}'.format(
                train_samples.shape[0]))
            logger.info('The size of development samples:{}'.format(
                dev_samples.shape[0]))
            logger.info('The size of testing samples:{}'.format(
                test_samples.shape[0]))

            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(
                save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(
                save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv(
                save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv(
                save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', index=None)


def gen_multi_step_forecast_samples(station, decomposer, lags_dict, columns, start, stop, test_len, wavelet_level="db10-2", lead_time=1,regen=False):
    """ 
    Generate multi-step training samples for autoregression problem. 
    This program could generate source CSV fflie for .tfrecords file generating. 
    Args:
        -station: The station where the original time series observed.
        -decomposer: The decomposition algorithm for decomposing the original time series.
        -lags_dict: The lags for autoregression.
        -columns: the columns name for read the source data by pandas
        -save_path: The path to save the training samples
    """
    logger.info(
        "Generating muliti-step decompositionensemble forecasting samples")
    logger.info('Station:{}'.format(station))
    logger.info('Decomposer:{}'.format(decomposer))
    logger.info('Lags_dict:{}'.format(lags_dict))
    logger.info('Signals:{}'.format(columns))
    logger.info('Validation start index:{}'.format(start))
    logger.info('Validation stop index:{}'.format(stop))
    logger.info('Testing sample length:{}'.format(test_len))
    logger.info(
        'Mother wavelet and decomposition level:{}'.format(wavelet_level))
    logger.info('Lead time:{}'.format(lead_time))
    if decomposer == "dwt" or decomposer == 'modwt':
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"
    save_path = data_path+"multi_step_"+str(lead_time)+"_ahead_forecast_pacf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path))>0 and not regen:
        logger.info('Learning samples have been generated!')
    else:
        logger.info("Save path:{}".format(save_path))
        # !!!!!!!!!!Generate training samples
        train_decompose_file = data_path+decomposer.upper()+"_TRAIN.csv"
        train_decompositions = pd.read_csv(train_decompose_file)
        train_decompositions.dropna()
        for k in range(len(columns)):
            lag = lags_dict[columns[k]]
            if lag == 0:
                logger.info("The lag of sub-signal({:.0f})".format(k+1)+" equals to 0")
                continue
            # Generate sample columns
            samples_columns = []
            for l in range(1, lag+1):
                samples_columns.append('X'+str(l))
            samples_columns.append('Y')
            # Obtain decomposed sub-signal
            sub_signal = train_decompositions[columns[k]]
            # convert pandas dataframe to numpy array
            nparr = np.array(sub_signal)
            # Create an empty pandas Dataframe
            train_samples = pd.DataFrame()
            # Generate input series based on lag and add these series to full dataset
            for i in range(lag):
                x = pd.DataFrame(
                    nparr[i:sub_signal.shape[0] - (lag - i)], columns=['X' + str(i + 1)])
                x = x.reset_index(drop=True)
                train_samples = pd.DataFrame(pd.concat([train_samples, x], axis=1))
            # Generate label data
            label = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])['Y']
            label = label.reset_index(drop=True)
            train_samples = train_samples[:train_samples.shape[0]-(lead_time-1)]
            train_samples = train_samples.reset_index(drop=True)
            # Add labled data to full_data_set
            train_samples = pd.concat([train_samples, label], axis=1, sort=False)
            # Do sampling if 'sampling' is True
            # Get the max and min value of each series
            series_max = train_samples.max(axis=0)
            series_min = train_samples.min(axis=0)
            # Normalize each series to the range between -1 and 1
            train_samples = 2 * (train_samples - series_min) / \
                (series_max - series_min) - 1
    
            # !!!!!Generate development and testing samples
            dev_test_samples = pd.DataFrame()
            appended_file_path = data_path+decomposer+"-test/"
            for j in range(start, stop+1):  # 遍历每一个附加分解结果
                data = pd.read_csv(appended_file_path+decomposer +
                                   '_appended_test'+str(j)+'.csv')
                imf = data[columns[k]]
                nparr = np.array(imf)
                inputs = pd.DataFrame()
                for i in range(lag):
                    x = pd.DataFrame(
                        nparr[i:nparr.size - (lag - i)], columns=['X' + str(i + 1)])
                    x = x.reset_index(drop=True)
                    inputs = pd.concat([inputs, x], axis=1, sort=False)
                label = pd.DataFrame(nparr[lag+lead_time-1:], columns=['Y'])['Y']
                label = label.reset_index(drop=True)
                inputs = inputs[:inputs.shape[0]-(lead_time-1)]
                inputs = inputs.reset_index(drop=True)
                full_data_set = pd.DataFrame(pd.concat([inputs, label], axis=1))
                last_imf = full_data_set.iloc[full_data_set.shape[0]-1:]
                dev_test_samples = pd.concat([dev_test_samples, last_imf], axis=0)
            dev_test_samples = dev_test_samples.reset_index(drop=True)
            dev_test_samples = 2*(dev_test_samples-series_min) / \
                (series_max-series_min)-1
            dev_samples = dev_test_samples.iloc[0:
                                                dev_test_samples.shape[0]-test_len]
            test_samples = dev_test_samples.iloc[dev_test_samples.shape[0]-test_len:]
            dev_samples = dev_samples.reset_index(drop=True)
            test_samples = test_samples.reset_index(drop=True)
    
            series_max = pd.DataFrame(series_max, columns=['series_max'])
            series_min = pd.DataFrame(series_min, columns=['series_min'])
            normalize_indicators = pd.concat([series_max, series_min], axis=1)
            normalize_indicators.to_csv(
                save_path+'norm_unsample_id_imf'+str(k+1)+'.csv')
            train_samples.to_csv(
                save_path+'minmax_unsample_train_imf'+str(k+1)+'.csv', index=None)
            dev_samples.to_csv(
                save_path+'minmax_unsample_dev_imf'+str(k+1)+'.csv', index=None)
            test_samples.to_csv(
                save_path+'minmax_unsample_test_imf'+str(k+1)+'.csv', index=None)
    