import sys
from variables import variables
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(root_path)
from samples_generator import gen_one_step_forecast_samples_triandev_test
from samples_generator import gen_multi_step_forecast_samples
from samples_generator import gen_one_step_forecast_samples


gen_one_step_forecast_samples_triandev_test(
    station="Huaxian",
    decomposer="eemd",
    lags_dict=variables['lags_dict'],
    input_columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
                   'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
    output_column=['ORIG'],
    start=673,
    stop=792,
    test_len=120,
)


gen_one_step_forecast_samples(
    station='Huaxian',
    decomposer='eemd',
    lags_dict=variables['lags_dict'],
    input_columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
                   'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
    output_column=['ORIG'],
    start=553,
    stop=792,
    test_len=120,
    mode='PACF',
    lead_time=1,
)

for lead_time in [3, 5, 7, 9]:
    gen_one_step_forecast_samples(
        station='Huaxian',
        decomposer='eemd',
        lags_dict=variables['lags_dict'],
        input_columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
                       'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
        output_column=['ORIG'],
        start=553,
        stop=792,
        test_len=120,
        mode='Pearson',
        lead_time=lead_time,
    )


gen_multi_step_forecast_samples(
    station='Huaxian',
    decomposer='eemd',
    lags_dict=variables['lags_dict'],
    columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
             'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
    start=553,
    stop=792,
    test_len=120,
)


gen_one_step_forecast_samples(
    station='Huaxian',
    decomposer='eemd',
    lags_dict=variables['lags_dict'],
    input_columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
                   'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
    output_column=['ORIG'],
    start=553,
    stop=792,
    test_len=120,
    mode='PACF',
    lead_time=1,
    n_components='mle',
)

num_in_one = sum(variables['lags_dict'].values())
for n_components in range(num_in_one-16, num_in_one+1):
    gen_one_step_forecast_samples(
        station='Huaxian',
        decomposer='eemd',
        lags_dict=variables['lags_dict'],
        input_columns=['IMF1', 'IMF2', 'IMF3', 'IMF4',
                       'IMF5', 'IMF6', 'IMF7', 'IMF8', 'IMF9'],
        output_column=['ORIG'],
        start=553,
        stop=792,
        test_len=120,
        mode='PACF',
        lead_time=1,
        n_components=n_components,
    )
