import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from PCA import one_step_pca

if __name__ == "__main__":
    one_step_pca(
        root_path=root_path,
        station='Zhangjiashan',
        decomposer='eemd',
        predict_pattern='forecast',
        n_components=54, # 54 input variables 
    )



