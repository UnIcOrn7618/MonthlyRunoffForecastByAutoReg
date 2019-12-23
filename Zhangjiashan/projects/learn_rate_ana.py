import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
# grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))

STATION='Zhangjiashan'
HIDDEN_UNITS = '8'
EPOCHS = '500'
BATCH_SIZE='512'
DROPOUT_RATE = '0.0'
DECAY_RATE = '0.0'
SEED_NUMBER = '1'

model_path = root_path+'/'+STATION+'/projects/lstm/'
graph_path = root_path+'/'+STATION+'/graph/'
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
def sort_key(s):
    if s:
        try:
            c = re.findall('LSTM-LR[\d+$]-HU['+HIDDEN_UNITS+']-EPS['+EPOCHS+']-BS['+BATCH_SIZE+']-DR['+DROPOUT_RATE+']-DC['+DECAY_RATE+']-SEED['+SEED_NUMBER+'].csv', s)[0]
        except:
            c = -1
        return int(c)
def strsort(alist):
    alist.sort(key=sort_key,reverse=True)
    return alist


plt.figure(figsize=(8,4))
pred_files_list=[]
for files in os.listdir(model_path):
    if files.find('HU['+HIDDEN_UNITS+']-EPS['+EPOCHS+']-BS['+BATCH_SIZE+']-DR['+DROPOUT_RATE+']-DC['+DECAY_RATE+']-SEED['+SEED_NUMBER+']-HISTORY')>0:
        pred_files_list.append(files)
pred_files_list=strsort(pred_files_list)


for i in range(0,len(pred_files_list)):
    # print(pred_files_list[i])
    data = pd.read_csv(model_path+pred_files_list[i])
    # print(data['loss'])
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.xlabel('Epochs', fontsize=10.5)
    plt.ylabel("MSE", fontsize=10.5)
    result = re.search('LR\\[(.*)\\]-HU',pred_files_list[i])
    lr=result.group(1)
    plt.plot(data['loss'],label='learning rate = '+lr)
    # plt.ylim([-0.0001,0.0025])
    plt.legend(
    # loc='upper left',
    loc=0,
    # bbox_to_anchor=(0.05,1),
    shadow=False,
    frameon=False,
    fontsize=10.5)
# plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.96, hspace=0.2, wspace=0.2)
plt.tight_layout()
plt.savefig(graph_path+'learn_rate_ana.eps', transparent=False, format='EPS', dpi=2000)
plt.show()


