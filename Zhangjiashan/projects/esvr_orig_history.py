import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from models import esvr,esvr_multi_seed


if __name__ == '__main__':
    
    # esvr(
    #     root_path=root_path,
    #     station='Zhangjiashan',
    #     n_calls=100,
    # )

    esvr_multi_seed(
        root_path=root_path,
        station='Zhangjiashan',
        n_calls=100,
    )
    
