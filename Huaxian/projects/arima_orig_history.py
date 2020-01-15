import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from models import arima
from variables import variables

if __name__ == '__main__':
    
    arima(
        root_path=root_path,
        station='Huaxian',
        variables=variables,
    )

    