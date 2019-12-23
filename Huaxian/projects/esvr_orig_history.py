import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path+'/tools/')
from models import esvr,multi_optimizer_esvr,esvr_multi_seed

if __name__ == '__main__':
    
    # esvr(
    #     root_path=root_path,
    #     station='Huaxian',
    #     n_calls=100,
    # )

    esvr_multi_seed(
        root_path=root_path,
        station='Huaxian',
        n_calls=100,
    )

    # multi_optimizer_esvr(
    #     root_path=root_path,
    #     station='Huaxian',
    #     n_calls=100,
    # )
    