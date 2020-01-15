import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from performance_analysis import pca_performance_ana

if __name__ == "__main__":
    pca_performance_ana(
        root_path=root_path,
        station='Huaxian',
        decomposer='eemd',
        predictor='esvr',
        start_n_components=25,
        stop_n_components=50,
    )