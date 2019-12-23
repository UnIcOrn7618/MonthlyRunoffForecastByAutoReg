import pandas as pd
import numpy as np
import math
from statistics import mean
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
from metrics_ import PPTS,mean_absolute_percentage_error

def read_two_stage(station,decomposer,predict_pattern,wavelet_level="db10-lev2"):
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"+predict_pattern+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+predict_pattern+"\\"
    predictions = pd.DataFrame()
    time_cost=[]
    for j in range(1,11):
        model_name = station+"_"+decomposer+"_esvr_"+predict_pattern+"_seed"+str(j)+".csv"
        data = pd.read_csv(model_path+model_name)
        if j==1:
            records = data['test_y'][0:120]
        test_pred=data['test_pred'][0:120]
        time_cost.append(data['time_cost'][0])
        test_pred=test_pred.reset_index(drop=True)
        predictions = pd.concat([predictions,test_pred],axis=1)
    predictions = predictions.mean(axis=1)
    records = records.values.flatten()
    predictions = predictions.values.flatten()
    r2=r2_score(y_true=records,y_pred=predictions)
    nrmse=math.sqrt(mean_squared_error(y_true=records,y_pred=predictions))/(sum(records)/len(predictions))
    mae=mean_absolute_error(y_true=records,y_pred=predictions)
    mape=mean_absolute_percentage_error(y_true=records,y_pred=predictions)
    ppts=PPTS(y_true=records,y_pred=predictions,gamma=5)
    time_cost=mean(time_cost)
    return records,predictions,r2,nrmse,mae,mape,ppts,time_cost

def read_two_stage_traindev_test(station,decomposer,predict_pattern,wavelet_level="db10-lev2"):
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"+predict_pattern+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+predict_pattern+"\\"
    test_predss = pd.DataFrame()
    dev_predss = pd.DataFrame()
    time_cost=[]
    for j in range(1,11):
        model_name = station+"_"+decomposer+"_esvr_"+predict_pattern+"_seed"+str(j)+".csv"
        data = pd.read_csv(model_path+model_name)
        if j==1:
            test_y = data['test_y'][0:120]
            dev_y = data['dev_y'][0:120]
        dev_pred=data['dev_pred'][0:120]
        test_pred=data['test_pred'][0:120]
        time_cost.append(data['time_cost'][0])
        dev_pred=dev_pred.reset_index(drop=True)
        test_pred=test_pred.reset_index(drop=True)
        test_predss = pd.concat([test_predss,test_pred],axis=1)
        dev_predss = pd.concat([dev_predss,dev_pred],axis=1)
    test_predss = test_predss.mean(axis=1)
    dev_predss = dev_predss.mean(axis=1)
    test_y = test_y.values.flatten()
    dev_y = dev_y.values.flatten()
    test_predss = test_predss.values.flatten()
    dev_predss = dev_predss.values.flatten()

    test_nse=r2_score(y_true=test_y,y_pred=test_predss)
    test_nrmse=math.sqrt(mean_squared_error(y_true=test_y,y_pred=test_predss))/(sum(test_y)/len(test_predss))
    test_mae=mean_absolute_error(y_true=test_y,y_pred=test_predss)
    test_mape=mean_absolute_percentage_error(y_true=test_y,y_pred=test_predss)
    test_ppts=PPTS(y_true=test_y,y_pred=test_predss,gamma=5)

    dev_nse=r2_score(y_true=dev_y,y_pred=dev_predss)
    dev_nrmse=math.sqrt(mean_squared_error(y_true=dev_y,y_pred=dev_predss))/(sum(dev_y)/len(dev_predss))
    dev_mae=mean_absolute_error(y_true=dev_y,y_pred=dev_predss)
    dev_mape=mean_absolute_percentage_error(y_true=dev_y,y_pred=dev_predss)
    dev_ppts=PPTS(y_true=dev_y,y_pred=dev_predss,gamma=5)

    metrics_dict={
        "dev_nse":dev_nse,
        "dev_nrmse":dev_nrmse,
        "dev_mae":dev_mae,
        "dev_mape":dev_mape,
        "dev_ppts":dev_ppts,
        "test_nse":test_nse,
        "test_nrmse":test_nrmse,
        "test_mae":test_mae,
        "test_mape":test_mape,
        "test_ppts":test_ppts,
        "time_cost":time_cost,
    }

    time_cost=mean(time_cost)
    return dev_y,dev_predss,test_y,test_predss,metrics_dict

def read_two_stage_max(station,decomposer,predict_pattern,wavelet_level="db10-lev2"):
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"+predict_pattern+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+predict_pattern+"\\"
    predictions = pd.DataFrame()
    time_cost=[]
    r2list=[]
    for j in range(1,11):
        model_name = station+"_"+decomposer+"_esvr_"+predict_pattern+"_seed"+str(j)+".csv"
        data = pd.read_csv(model_path+model_name)
        r2list.append(data['test_r2'][0])
    print("one-month NSE LIST:{}".format(r2list))
    max_id = r2list.index(max(r2list))
    print("one-month max id:{}".format(max_id))
    model_name = station+"_"+decomposer+"_esvr_"+predict_pattern+"_seed"+str(max_id+1)+".csv"
    data = pd.read_csv(model_path+model_name)    
    records = data['test_y'][0:120]
    test_pred=data['test_pred'][0:120]
    records = records.values.flatten()
    predictions = test_pred.values.flatten()
    r2=data['test_r2'][0]
    nrmse=data['test_nrmse'][0]
    mae=data['test_mae'][0]
    mape=data['test_mape'][0]
    ppts=data['test_ppts'][0]
    time_cost=data['time_cost'][0] 
    return records,predictions,r2,nrmse,mae,mape,ppts,time_cost

def read_pure_esvr(station):
    model_path = root_path+"\\"+station+"\\projects\\esvr\\"
    predictions = pd.DataFrame()
    time_cost=[]
    for j in range(1,11):
        model_name = station+"_esvr_seed"+str(j)+".csv"
        data = pd.read_csv(model_path+model_name)
        if j==1:
            records = data['test_y'][0:120]
        test_pred=data['test_pred'][0:120]
        time_cost.append(data['time_cost'][0])
        test_pred=test_pred.reset_index(drop=True)
        predictions = pd.concat([predictions,test_pred],axis=1)
    predictions = predictions.mean(axis=1)
    records = records.values.flatten()
    predictions = predictions.values.flatten()
    r2=r2_score(y_true=records,y_pred=predictions)
    nrmse=math.sqrt(mean_squared_error(y_true=records,y_pred=predictions))/(sum(records)/len(records))
    mae=mean_absolute_error(y_true=records,y_pred=predictions)
    mape=mean_absolute_percentage_error(y_true=records,y_pred=predictions)
    ppts=PPTS(y_true=records,y_pred=predictions,gamma=5)
    time_cost=mean(time_cost)
    return records,predictions,r2,nrmse,mae,mape,ppts,time_cost


def read_pca_metrics(station,decomposer,start_component,stop_component,wavelet_level="db10-lev2"):
    
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\data\\"+wavelet_level+"\\one_step_1_month_forecast\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\data\\one_step_1_month_forecast\\"
    train = pd.read_csv(model_path+"minmax_unsample_train.csv")
    dev = pd.read_csv(model_path+"minmax_unsample_dev.csv")
    test = pd.read_csv(model_path+"minmax_unsample_test.csv")
    norm_id=pd.read_csv(model_path+"norm_unsample_id.csv")
    sMax = (norm_id['series_max']).values
    sMin = (norm_id['series_min']).values
    # Conncat the training, development and testing samples
    samples = pd.concat([train,dev,test],axis=0)
    samples = samples.reset_index(drop=True)
    # Renormalized the entire samples
    samples = np.multiply(samples + 1,sMax - sMin) / 2 + sMin
    y = samples['Y']
    X = samples.drop('Y',axis=1)
    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_
    print("n_components_pca_mle:{}".format(n_components_pca_mle))
    mle = X.shape[1]-n_components_pca_mle

    nrmse=[]
    r2=[]
    mae=[]
    mape=[]
    ppts=[]
    for i in range(start_component,stop_component+1):
        if decomposer=="wd":
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\one_step_1_month_forecast_with_pca_"+str(i)+"\\"
        else:
            model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\one_step_1_month_forecast_with_pca_"+str(i)+"\\"
        # averaging the trained svr with different seed
        test_pred_df = pd.DataFrame()
        for j in range(1,11):
            model_name = station+"_"+decomposer+"_esvr_one_step_1_month_forecast_with_pca_"+str(i)+"_seed"+str(j)+".csv"
            data = pd.read_csv(model_path+model_name)
            
            test_y = data['test_y'][0:120]
            test_pred=data['test_pred'][0:120]
            test_pred_df = pd.concat([test_pred_df,test_pred],axis=1)
        test_pred = test_pred_df.mean(axis=1)
        test_y = test_y.values
        test_pred = test_pred.values
        print(type(test_y))
        print(type(test_pred))
        r2.append(r2_score(y_true=test_y,y_pred=test_pred))
        nrmse.append(math.sqrt(mean_squared_error(y_true=test_y,y_pred=test_pred))/(sum(test_y)/len(test_y)))
        mae.append(mean_absolute_error(y_true=test_y,y_pred=test_pred))
        mape.append(mean_absolute_percentage_error(y_true=test_y,y_pred=test_pred))
        ppts.append(PPTS(y_true=test_y,y_pred=test_pred,gamma=5))

    pc0_records,pc0_predictions,pc0_r2,pc0_nrmse,pc0_mae,pc0_mape,pc0_ppts,pc0_time_cost=read_two_stage(station=station,decomposer=decomposer,predict_pattern="one_step_1_month_forecast",)

    r2.append(pc0_r2)
    nrmse.append(pc0_nrmse)
    mae.append(pc0_mae)
    mape.append(pc0_mape)
    ppts.append(pc0_ppts)

    r2.reverse()
    nrmse.reverse()
    mae.reverse()
    mape.reverse()
    ppts.reverse()

    return mle,r2,nrmse,mae,mape,ppts

def read_long_leading_time(station,decomposer,mode='new',wavelet_level="db10-lev2"):
    records=[]
    predictions=[]
    nrmse=[]
    r2=[]
    mae=[]
    mape=[]
    ppts=[]
    
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"

    m1_records,m1_predictions,m1_r2,m1_nrmse,m1_mae,m1_mape,m1_ppts,m1_time_cost=read_two_stage(station=station,decomposer=decomposer,predict_pattern="one_step_1_month_forecast",)
    records.append(m1_records)
    predictions.append(m1_predictions)
    r2.append(m1_r2)
    nrmse.append(m1_nrmse)
    mae.append(m1_mae)
    mape.append(m1_mape)
    ppts.append(m1_ppts)
    # averaging the trained svr with different seed
    test_pred_df = pd.DataFrame()
    leading_times=[3,5,7,9]
    for leading_time in leading_times:
        
        print("Reading  mode:{}".format(mode))
        if mode==None:
            file_path = model_path+"one_step_"+str(leading_time)+"_month_forecast//"
        else:
            file_path = model_path+"one_step_"+str(leading_time)+"_month_forecast_"+mode+"//"
        
        for j in range(1,11):
            if mode == None:
                model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_seed"+str(j)+".csv"
            else:
                model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_"+mode+"_seed"+str(j)+".csv"
            data = pd.read_csv(file_path+model_name)
            test_y = data['test_y'][0:120]
            test_pred=data['test_pred'][0:120]
            test_pred_df = pd.concat([test_pred_df,test_pred],axis=1)
        test_pred = test_pred_df.mean(axis=1)
        test_y = test_y.values
        test_pred = test_pred.values
        print(type(test_y))
        print(type(test_pred))
        records.append(test_y)
        predictions.append(test_pred)
        r2.append(r2_score(y_true=test_y,y_pred=test_pred))
        nrmse.append(math.sqrt(mean_squared_error(y_true=test_y,y_pred=test_pred))/(sum(test_y)/len(test_y)))
        mae.append(mean_absolute_error(y_true=test_y,y_pred=test_pred))
        mape.append(mean_absolute_percentage_error(y_true=test_y,y_pred=test_pred))
        ppts.append(PPTS(y_true=test_y,y_pred=test_pred,gamma=5))

    return records,predictions,r2,nrmse,mae,mape,ppts


def read_long_leading_time_max(station,decomposer,model='new',wavelet_level="db10-lev2"):
    records=[]
    predictions=[]
    nrmse=[]
    r2=[]
    mae=[]
    mape=[]
    ppts=[]
    
    if decomposer=="wd":
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"+wavelet_level+"\\"
    else:
        model_path = root_path+"\\"+station+"_"+decomposer+"\\projects\\esvr\\"

    m1_records,m1_predictions,m1_r2,m1_nrmse,m1_mae,m1_mape,m1_ppts,m1_time_cost=read_two_stage_max(station=station,decomposer=decomposer,predict_pattern="one_step_1_month_forecast",)
    records.append(m1_records)
    predictions.append(m1_predictions)
    r2.append(m1_r2)
    nrmse.append(m1_nrmse)
    mae.append(m1_mae)
    mape.append(m1_mape)
    ppts.append(m1_ppts)
    # averaging the trained svr with different seed
    test_pred_df = pd.DataFrame()
    leading_times=[3,5,7,9]
    for leading_time in leading_times:
        if model=='new':
            print("Reading new model"+"."*100)
            file_path = model_path+"one_step_"+str(leading_time)+"_month_forecast_new//"
        else:
            file_path = model_path+"one_step_"+str(leading_time)+"_month_forecast//"
        r2list=[]
        for j in range(1,11):
            if model=='new':
                model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_new_seed"+str(j)+".csv"
            else:
                model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_seed"+str(j)+".csv"
            data = pd.read_csv(file_path+model_name)
            r2list.append(data['test_r2'][0])
        print("NSE LIST:{}".format(r2list))
        max_id = r2list.index(max(r2list))
        print("max id:{}".format(max_id))
        if model=='new':
            model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_new_seed"+str(max_id+1)+".csv"
        else:
            model_name = station+"_"+decomposer+"_esvr_one_step_"+str(leading_time)+"_month_forecast_seed"+str(max_id+1)+".csv"
        data = pd.read_csv(file_path+model_name)
        r2.append(data['test_r2'][0])
        nrmse.append(data['test_nrmse'][0])
        mae.append(data['test_mae'][0])
        mape.append(data['test_mape'][0])
        ppts.append(data['test_ppts'][0])
        test_y = data['test_y'][0:120]
        test_pred=data['test_pred'][0:120]
        test_y = test_y.values
        test_pred = test_pred.values
        print(type(test_y))
        print(type(test_pred))
        records.append(test_y)
        predictions.append(test_pred)
    return records,predictions,r2,nrmse,mae,mape,ppts


def read_samples_num(station,decomposer,pre=20,wavelet_level="db10-lev2"):
    if decomposer=="wd":
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"+wavelet_level+"/"
    else:
        data_path = root_path+"/"+station+"_"+decomposer+"/data/"

    leading_time=[3,5,7,9]
    thresh=[0.1,0.2,0.3,0.4,0.5]
    num_sampless=[]
    for lt in leading_time:
        num_samples=[]
        for t in thresh:
            data = pd.read_csv(data_path+"one_step_"+str(lt)+"_month_forecast_pre"+str(pre)+"_thresh"+str(t)+"/minmax_unsample_train.csv")
            data.drop("Y",axis=1)
            num_samples.append(data.shape[1])
        num_sampless.append(num_samples)
    return num_sampless