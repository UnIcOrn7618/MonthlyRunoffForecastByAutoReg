import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path+'/tools/')
from PCA import one_step_pca

if __name__ == "__main__":
    # for n_components in range(40,41):
    one_step_pca(
        root_path=root_path,
        station='Huaxian',
        decomposer='ssa',
        predict_pattern='forecast',
        n_components=55, #55 input variables
    )



