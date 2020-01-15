import os
root_path = os.path.dirname(os.path.abspath('__file__'))

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

if __name__ == "__main__":
    # 严重，不要执行此文件
    del_file(root_path+'/Huaxian/projects/esvr/')
    del_file(root_path+'/Huaxian_eemd/projects/esvr/')
    del_file(root_path+'/Huaxian_ssa/projects/esvr/')
    del_file(root_path+'/Huaxian_vmd/projects/esvr/')
    del_file(root_path+'/Huaxian_dwt/projects/esvr/')
    del_file(root_path+'/Xianyang/projects/esvr/')
    del_file(root_path+'/Xianyang_eemd/projects/esvr/')
    del_file(root_path+'/Xianyang_ssa/projects/esvr/')
    del_file(root_path+'/Xianyang_vmd/projects/esvr/')
    del_file(root_path+'/Xianyang_dwt/projects/esvr/')
    del_file(root_path+'/Zhangjiashan/projects/esvr/')
    del_file(root_path+'/Zhangjiashan_eemd/projects/esvr/')
    del_file(root_path+'/Zhangjiashan_ssa/projects/esvr/')
    del_file(root_path+'/Zhangjiashan_vmd/projects/esvr/')
    del_file(root_path+'/Zhangjiashan_dwt/projects/esvr/')