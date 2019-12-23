#%%
import os
root_path = os.path.dirname(os.path.abspath("__file__"))
from Xianyang.projects.variables import lags
from tools.samples_generator import gen_samples_minmax



#%%
# Generate samples for Xianyang
gen_samples_minmax(
    source_path=root_path+'/time_series/XianyangRunoff1951-2018(1953-2018).xlsx',
    lag = lags[0],
    column=['MonthlyRunoff'],
    save_path=root_path+'/Xianyang/data/',
    val_len=120,
    seed=20190610,
    sampling=False,
    header=True,
    index=False,
)


#%%
