import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
# def rename_dict(dir_path,spliter=None,replace=None):
#     print(dir_path)
#     # print(os.listdir(dir_path))
#     for x in os.listdir(dir_path):
#         if x.__contains__("_one_step_esvr_"):
#             alist = x.split("_one_step_esvr_")
#             new_name = alist[0]+"_esvr_one_step_1_month_"+alist[1]
#             print(new_name)
#             os.rename(dir_path+"/"+x,dir_path+"/"+new_name)

def rename_dict(dir_path,spliter=None,replace=None):
    print(dir_path)
    # print(os.listdir(dir_path))
    for x in os.listdir(dir_path):
        if x.__contains__(spliter):
            print("Original name:{}".format(x))
            alist = x.split(spliter)
            new_name = alist[0]+replace+alist[1]
            print("New name:{}".format(new_name))
            os.rename(dir_path+"/"+x,dir_path+"/"+new_name)

    
if __name__ == "__main__":
    # for i in range(1,8):
    #     rename_dict(
    #         dir_path=root_path+"/Zhangjiashan_vmd/projects/esvr/multi_step_1_month_forecast/imf"+str(i)+"/",
    #         spliter="_multi_step_esvr_",
    #         replace="_esvr_multi_step_1_month_",
    #     )
    dir_path = root_path+'/Huaxian_eemd/projects/esvr/'
    for x in os.listdir(dir_path):
        if x.__contains__('pca') and 'ahead' not in x and 'metrics' not in x:
            print(x)
            alist = x.split('_')
            new_name = alist[0]+'_'+alist[1]+'_'+alist[2]+'_ahead_'+alist[4]+'_pacf_'+alist[6]+alist[7]
            print(new_name)
            os.rename(dir_path+"/"+x,dir_path+"/"+new_name)
        elif x.__contains__('multi_step_1_month'):
            print(x)
            alist = x.split('month')
            new_name = alist[0]+'ahead'+alist[1]+'_pacf'
            print(new_name)
            os.rename(dir_path+"/"+x,dir_path+"/"+new_name)
        elif x.__contains__('one_step') and x.__contains__('new'):
            print(x)
            alist = x.split('_')
            new_name = alist[0]+'_'+alist[1]+'_'+alist[2]+'_ahead_'+alist[4]+'_pacf'
            print(new_name)
            os.rename(dir_path+"/"+x,dir_path+"/"+new_name)
        # elif x.__contains__('one_step') and 'pca' not in x:
        #     print(x)
        #     alist = x.split('month')
        #     new_name = alist[0]+'ahead'+alist[1]+'_pacf'
        #     print(new_name)
        #     os.rename(dir_path+"/"+x,dir_path+"/"+new_name)
        

        