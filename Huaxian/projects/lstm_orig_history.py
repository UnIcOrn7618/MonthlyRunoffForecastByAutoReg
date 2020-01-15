import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
root_path = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root_path)
from models import lstm

if __name__ == "__main__":

    lstm(
        root_path=root_path,
        station='Huaxian',
        seed=1,
        epochs_num=5000,
        batch_size=128,
        learning_rate=0.007,
        decay_rate=0.0,
        hidden_layer=1,
        hidden_units_1=8,
        dropout_rate_1=0.0,
        hidden_units_2=8,
        dropout_rate_2=0.0,
        early_stoping=True,
        retrain=False,
        warm_up=False,
        initial_epoch=None,
    )
    plt.show()