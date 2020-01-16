from variables import variables
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
import sys
sys.path.append(root_path)
from tools.samples_generator import generate_monoscale_samples

generate_monoscale_samples(
    source_file=root_path +
    '/time_series/ZhangjiashanRunoff1953-2018(1953-2018).xlsx',
    save_path=root_path+'/Zhangjiashan/data/',
    lags_dict=variables['lags_dict'],
    column=['MonthlyRunoff'],
    test_len=120,
)
