import numpy as np
import pandas as pd
import math
import os
root_path = os.path.dirname(os.path.abspath('__file__'))

def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def PPTS(y_true,y_pred,gamma):
    """ 
    Compute peak percentage threshold statistic
    args:
        y_true:the observed records
        y_pred:the predictions
        gamma:lower value percentage
    """
    # y_true
    r = pd.DataFrame(y_true,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(y_pred,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=False)
    rps_g=rps.iloc[:G]
    y_true = (rps_g['r']).values
    y_pred = (rps_g['p']).values
    abss=np.abs((y_true-y_pred)/y_true*100)
    print('abs error={}'.format(abss))
    sums = np.sum(abss)
    print('sum of abs error={}'.format(abss))
    ppts = sums*(1/((gamma/100)*N))
    print('ppts('+str(gamma)+'%)={}'.format(ppts))
    return ppts

def r2_score(y_true,y_pred):
    y_true_avg=sum(y_true)/len(y_true)
    y_pred_avg=sum(y_pred)/len(y_pred)
    r2=sum((y_true-y_true_avg)*(y_pred-y_pred_avg))/math.sqrt(sum((y_true-y_true_avg)**2)*sum((y_pred-y_pred_avg)**2))
    return r2

if __name__ == '__main__':
    data=pd.read_csv(root_path+"/Huaxian_vmd/projects/esvr/multi_step_1_month_forecast/esvr_Huaxian_vmd_sum_test_result.csv")

    print(data)
    y_true = data['orig']
    y_pred = data['pred']
    r2 = r2_score(y_true=y_true,y_pred=y_pred)
    print(r2)

    # data = pd.read_excel(root_path+'\\e-svr-models\\gbr_pred_e_svr.xlsx')
    # # test_data_size=4325
    # y_test = data['y_train'][1:test_data_size+1]
    # test_predictions=data['train_pred'][1:test_data_size+1]
    # # print(y_test)
    # ppts = PPTS(y_test.values,test_predictions.values,5)
    # # print(ppts)

    # data = pd.read_excel(root_path+'\\bpnn-models\\gbr_pred_bpnn.xlsx')
    # # test_data_size=4325
    # y_test = data['y_train'][1:test_data_size+1]
    # test_predictions=data['train_pred'][1:test_data_size+1]
    # # print(y_test)
    # ppts = PPTS(y_test.values,test_predictions.values,5)
    # # print(ppts)
    # print('='*100)
    # data = pd.read_csv(root_path+'\\lstm-models\\lstm_ensemble_test_result.csv')
    # # test_data_size=4325
    # y_test = data['orig']
    # test_predictions=data['pred']
    # # print(y_test)
    # ppts5 = PPTS(y_test.values,test_predictions.values,5)
    # # print('ppts5={}'.format(ppts5))
    # ppts15 = PPTS(y_test.values,test_predictions.values,15)
    # # print('ppts15={}'.format(ppts15))
    # ppts20 = PPTS(y_test.values,test_predictions.values,20)
    # # print('ppts20={}'.format(ppts20))
    # ppts25 = PPTS(y_test.values,test_predictions.values,25)
    # # print('ppts25={}'.format(ppts25))
