import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from models import gbrt


if __name__ == '__main__':
    gbrt(
        root_path=root_path,
        station='Zhangjiashan',
        n_calls=100,
    )
    plt.show()


    
