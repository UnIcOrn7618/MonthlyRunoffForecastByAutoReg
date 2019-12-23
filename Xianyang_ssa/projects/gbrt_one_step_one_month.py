import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path+'/tools/')
from models import one_step_gbrt


if __name__ == '__main__':
    one_step_gbrt(
        root_path=root_path,
        station='Xianyang',
        decomposer='ssa',
        predict_pattern='forecast'
    )
    plt.show()


    
