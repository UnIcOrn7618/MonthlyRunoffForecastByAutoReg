import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
print("Root path:{}".format(root_path))
import sys
sys.path.append(root_path+'/tools/')
from PCA import one_step_pca

if __name__ == "__main__":
    one_step_pca(
        root_path=root_path,
        station='Huaxian',
        decomposer='eemd',
        predict_pattern='forecast',
        n_components= 51, # 51 input variables
    )



