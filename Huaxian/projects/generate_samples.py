import os
root_path = os.path.dirname(os.path.abspath("__file__"))
from variables import variables
import sys
sys.path.append(root_path)
from tools.samples_generator import generate_monoscale_samples


# Generate samples for Huaxian
generate_monoscale_samples(
    source_file=root_path+'/time_series/HuaxianRunoff1951-2018(1953-2018).xlsx',
    save_path=root_path+'/Huaxian/data/',
    lags_dict = variables['lags_dict'],
    column=['MonthlyRunoff'],
    test_len=120,
)
