import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=9
plt.rcParams["figure.figsize"] = [7.48, 5.61]


def pca_performance_ana(root_path,station,decomposer,predictor,start_n_components,stop_n_components,predict_pattern='forecast'):
    models_path = root_path+'/'+station+'_'+decomposer+'/projects/'+predictor+'/'
    n_components = []
    train_rmse=[]
    train_r2 = []
    train_mae = []
    train_mape = []
    train_ppts = []
    dev_rmse = []
    dev_r2 = []
    dev_mae = []
    dev_mape = []
    dev_ppts = []
    test_rmse = []
    test_r2 = []
    test_mae = []
    test_mape = []
    test_ppts = []
    for i in range (start_n_components,stop_n_components+1):
        n_components.append(i)
        tr_rmse=[]
        tr_r2=[]
        tr_mae=[]
        tr_mape=[]
        tr_ppts=[]
        de_rmse=[]
        de_r2=[]
        de_mae=[]
        de_mape=[]
        de_ppts=[]
        te_rmse=[]
        te_r2=[]
        te_mae=[]
        te_mape=[]
        te_ppts=[]
        for j in range(1,11):
            model_name = station+'_'+decomposer+'_one_step_'+predictor+'_'+predict_pattern+'_with_pca_'+str(i)+'_seed'+str(j)+'.csv'
            model = models_path+'one_step_one_month_'+predict_pattern+'_with_pca_'+str(i)+'/'+model_name
            data = pd.read_csv(model)
            tr_rmse.append(data['train_rmse'][0])
            tr_r2.append(data['train_r2'][0])
            tr_mae.append(data['train_mae'][0])
            tr_mape.append(data['train_mape'][0])
            tr_ppts.append(data['train_ppts'][0])
            de_rmse.append(data['dev_rmse'][0])
            de_r2.append(data['dev_r2'][0])
            de_mae.append(data['dev_mae'][0])
            de_mape.append(data['dev_mape'][0])
            de_ppts.append(data['dev_ppts'][0])
            te_rmse.append(data['test_rmse'][0])
            te_r2.append(data['test_r2'][0])
            te_mae.append(data['test_mae'][0])
            te_mape.append(data['test_mape'][0])
            te_ppts.append(data['test_ppts'][0])
        train_rmse.append(sum(tr_rmse)/10)
        train_r2.append(sum(tr_r2)/10)
        train_mae.append(sum(tr_mae)/10)
        train_mape.append(sum(tr_mape)/10)
        train_ppts.append(sum(tr_ppts)/10)
        dev_rmse.append(sum(de_rmse)/10)
        dev_r2.append(sum(de_r2)/10)
        dev_mae.append(sum(de_mae)/10)
        dev_mape.append(sum(de_mape)/10)
        dev_ppts.append(sum(de_ppts)/10)
        test_rmse.append(sum(te_rmse)/10)
        test_r2.append(sum(te_r2)/10)
        test_mae.append(sum(te_mae)/10)
        test_mape.append(sum(te_mape)/10)
        test_ppts.append(sum(te_ppts)/10)

    pca_models_metrics={
        'n_components':n_components,
        'train_rmse':train_rmse,
        'train_r2':train_r2,
        'train_mae':train_mae,
        'train_mape':train_mape,
        'train_ppts':train_ppts,

        'dev_rmse':dev_rmse,
        'dev_r2':dev_r2,
        'dev_mae':dev_mae,
        'dev_mape':dev_mape,
        'dev_ppts':dev_ppts,

        'test_rmse':test_rmse,
        'test_r2':test_r2,
        'test_mae':test_mae,
        'test_mape':test_mape,
        'test_ppts':test_ppts,

    }

    pca_models_metrics_df = pd.DataFrame(pca_models_metrics)
    pca_models_metrics_df.to_csv(models_path+'pca_models_metrics.csv')
    print(id(test_rmse))
    test_rmse.reverse()
    print(id(test_rmse))
    print(test_rmse)

    plt.figure()
    plt.xlabel('Reduced number of dimensions')
    plt.ylabel('RMSE')
    plt.bar(x=list(range(1,len(n_components)+1)),height=test_rmse)
    plt.tight_layout()
    plt.show()


