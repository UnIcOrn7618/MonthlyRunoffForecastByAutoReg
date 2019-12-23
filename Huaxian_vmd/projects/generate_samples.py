import os
root_path = os.path.dirname(os.path.abspath("__file__"))
from variables import one_step_lags,multi_step_lags
print("One-Step Lags:{}".format(one_step_lags))
print("Multi-Step Lags:{}".format(multi_step_lags))
import sys
sys.path.append(root_path+'/tools/')
from samples_generator import gen_one_step_hindcast_samples
from samples_generator import gen_one_step_forecast_samples
from samples_generator import gen_multi_step_hindcast_samples
from samples_generator import gen_multi_step_forecast_samples
from samples_generator import gen_one_step_forecast_samples_leading_time
from samples_generator import gen_one_step_forecast_samples_triandev_test

# Generate one-step one-month ahead hindcast samples
# gen_one_step_hindcast_samples(
#     station='Huaxian',
#     decomposer="vmd",
#     lags = one_step_lags,
#     input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
#     output_column=['ORIG'],
#     test_len=120,
#     seed=20190610,
#     sampling=False,
#     header=True,
#     index=False,
# )

# gen_one_step_forecast_samples_leading_time(
#         station='Huaxian',
#         decomposer='vmd',
#         lags=one_step_lags,
#         input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
#         output_column=['ORIG'],
#         start=553,
#         stop=792,
#         test_len=120,
#         leading_time=9,
#     )


gen_one_step_forecast_samples_triandev_test(
        station="Huaxian",
        decomposer="vmd",
        lags=one_step_lags,
        input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
        output_column=['ORIG'],
        start=673,
        stop=792,
        test_len=120,
    )

# Generate one-step one-month ahead forecast samples
# gen_one_step_forecast_samples(
#     station = "Huaxian",
#     decomposer="vmd",
#     lags=one_step_lags,
#     input_columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
#     output_column=['ORIG'],
#     start=553,
#     stop=792,
#     test_len=120,
#     seed=20190610,
#     sampling=False,
#     header=True,
#     index=False,
# )


# Generate multi-step one-month ahead hindcast samples
# gen_multi_step_hindcast_samples(
#     station='Huaxian',
#     decomposer='vmd',
#     lags=multi_step_lags,
#     columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
#     test_len=120,
#     sampling=False,
#     header=True,
#     index=False, 
# )

# Generate multi-step one-month ahead forecast samples
# gen_multi_step_forecast_samples(
#     station='Huaxian',
#     decomposer="vmd",
#     lags=multi_step_lags,
#     columns=['IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8',],
#     start=553,
#     stop=792,
#     test_len=120,
#     sampling=False,
#     header=True,
#     index=False
# )

