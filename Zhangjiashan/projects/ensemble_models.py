import matplotlib.pyplot as plt
import os
root_path = os.path.dirname(os.path.abspath('__file__')) 
import sys
sys.path.append(root_path)
from tools.ensembler import ensemble
from Zhangjiashan.projects.variables import variables

# Set the project parameters
ORIGINAL = 'ZhangjiashanRunoff1953-2018(1953-2018).xlsx'
STATION = 'Zhangjiashan'
PREDICTOR = 'esvr' # esvr or gbrt or lstm
PREDICT_PATTERN = '1_ahead_pacf'

ensemble(
    root_path=root_path,
    original_series=ORIGINAL,
    station=STATION,
    variables = variables,
    predictor=PREDICTOR,
    predict_pattern=PREDICT_PATTERN,
)
plt.show()
