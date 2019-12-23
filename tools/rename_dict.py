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

    files=os.listdir(root_path)
    print(len(files))