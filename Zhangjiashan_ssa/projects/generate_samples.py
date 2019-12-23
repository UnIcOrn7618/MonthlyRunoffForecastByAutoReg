import sys
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(root_path+'/tools/')
from variables import one_step_lags, multi_step_lags
from samples_generator import gen_one_step_forecast_samples_leading_time
from samples_generator import gen_multi_step_forecast_samples
from samples_generator import gen_multi_step_hindcast_samples
from samples_generator import gen_one_step_forecast_samples
from samples_generator import gen_one_step_hindcast_samples
from samples_generator import gen_one_step_forecast_samples_triandev_test
print("One-Step Lags:{}".format(one_step_lags))
print("Multi-Step Lags:{}".format(multi_step_lags))
sys.path.append(root_path+'/tools/')


# Generate one-step one-month ahead hindcast samples
# gen_one_step_hindcast_samples(
#     station='Zhangjiashan',
#     decomposer="ssa",
#     lags = one_step_lags,
#     input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
#     output_column=['ORIG'],
#     test_len=120,
#     seed=20190610,
#     sampling=False,
#     header=True,
#     index=False,
# )

# gen_one_step_forecast_samples_leading_time(
#         station='Zhangjiashan',
#         decomposer='ssa',
#         lags=one_step_lags,
#         input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
#         output_column=['ORIG'],
#         start=553,
#         stop=792,
#         test_len=120,
#         leading_time=9,
#     )

gen_one_step_forecast_samples_triandev_test(
    station="Zhangjiashan",
    decomposer="ssa",
    lags=one_step_lags,
    input_columns=['Trend', 'Periodic1', 'Periodic2', 'Periodic3', 'Periodic4', 'Periodic5',
                   'Periodic6', 'Periodic7', 'Periodic8', 'Periodic9', 'Periodic10', 'Noise'],
    output_column=['ORIG'],
    start=673,
    stop=792,
    test_len=120,
)

# Generate one-step one-month ahead forecast samples
# gen_one_step_forecast_samples(
#     station = 'Zhangjiashan',
#     decomposer="ssa",
#     lags=one_step_lags,
#     input_columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
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
#     station='Zhangjiashan',
#     decomposer='ssa',
#     lags=multi_step_lags,
#     columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
#     test_len=120,
#     sampling=False,
#     header=True,
#     index=False,
# )


# Generate multi-step one-month ahead forecast samples
# gen_multi_step_forecast_samples(
#     station='Zhangjiashan',
#     decomposer="ssa",
#     lags=multi_step_lags,
#     columns=['Trend','Periodic1','Periodic2','Periodic3','Periodic4','Periodic5','Periodic6','Periodic7','Periodic8','Periodic9','Periodic10','Noise'],
#     start=553,
#     stop=792,
#     test_len=120,
#     sampling=False,
#     header=True,
#     index=False
# )
