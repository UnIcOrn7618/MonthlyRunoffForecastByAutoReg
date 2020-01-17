close all;
clear
clc

stations={"Huaxian","Xianyang","Zhangjiashan",};
decomposers={"eemd","vmd","dwt","modwt","ssa"};

for i =1:length(stations)
    station=string(stations(i));
    switch station
        case 'Huaxian'
            orig = xlsread("../time_series/HuaxianRunoff1951-2018(1953-2018).xlsx",1,'B26:B817');%full=792sample
        case 'Xianyang'
            orig = xlsread("../time_series/XianyangRunoff1951-2018(1953-2018).xlsx",1,'B26:B817');%full=792samples
        case 'Zhangjiashan'
            orig = xlsread("../time_series/ZhangJiaShanRunoff1953-2018(1953-2018).xlsx",1,'B2:B793');%full=792samples
    end
    orig_train = orig(1:552);
    NumLags=20;
    up_bounds=zeros(NumLags+1,1);
    lo_bounds=zeros(NumLags+1,1);
    [pacf,lags,bounds] = parcorr(orig_train,'NumLags',NumLags);
    up_bounds(:,1)=bounds(1);
    lo_bounds(:,1)=bounds(2);
    PACF_DATA=[pacf,up_bounds,lo_bounds];
    PACF_TABLE = array2table(PACF_DATA, 'VariableNames', {'ORIG','UP','LOW'});
    save_path = strcat("../",station,"/data/")
    if exist(save_path,'dir')==0
            mkdir(save_path);
    end
    writetable(PACF_TABLE, strcat(save_path,"/PACF.csv"));
   
    
    for j =1:length(decomposers)
        decomposer=string(decomposers(j));
        if decomposer== "dwt" || decomposer=="modwt"
            decomposition_file=strcat("../",station,"_",decomposer,"/data/db10-2/",upper(decomposer),"_TRAIN.csv");
        else
            decomposition_file=strcat("../",station,"_",decomposer,"/data/",upper(decomposer),"_TRAIN.csv");
        end
        data = readtable(decomposition_file);
        [m,n]=size(data);
        columns = {};
        for k=1:n+2
            if decomposer=="eemd" || decomposer=="vmd"
                if k==1
                    columns{k}='ORIG';
                else
                    columns{k}=['IMF',num2str(k-1)];
                end
            elseif decomposer=="dwt" || decomposer=="modwt"
                if k==1
                    columns{k}='ORIG';
                elseif k==n
                    columns{k}=['A',num2str(n-2)];
                else
                    columns{k}=['D',num2str(k-1)];
                end
            else
                if k==1
                    columns{k}='ORIG';
                elseif k==2
                    columns{k}=['Trend'];
                elseif k==n
                    columns{k}=['Noise'];
                else
                    columns{k}=['Periodic',num2str(k-2)];
                end
            end
            if k==n+1
                columns{k}='UP';
            elseif k==n+2
                columns{k}='LOW';
            end
        end
        
        pacfs=zeros(NumLags+1,n);
        up_bounds=zeros(NumLags+1,1);
        lo_bounds=zeros(NumLags+1,1);
        for k=1:n
            eval(strcat('signal','=','data.',columns{k},';'))
            [pacf,lags,bounds] = parcorr(signal,'NumLags',NumLags);
            pacfs(:,k)=pacf;
            if k==1
                up_bounds(:,1)=bounds(1);
                lo_bounds(:,1)=bounds(2);
            end
        end
        PACF_DATA=[pacfs,up_bounds,lo_bounds];
        PACF_TABLE = array2table(PACF_DATA, 'VariableNames', columns);
        
        if decomposer=="dwt" || decomposer=="modwt"
            save_path = strcat("../",station,"_",decomposer,"/data/db10-2/")
%             writetable(PACF_TABLE, strcat("../",station,"_",decomposer,"/data/db10-2/PACF.csv"));
        else
%             writetable(PACF_TABLE, strcat("../",station,"_",decomposer,"/data/PACF.csv"));
            save_path = strcat("../",station,"_",decomposer,"/data/")
        end
        if exist(save_path,'dir')==0
            mkdir(save_path);
        end
        writetable(PACF_TABLE, strcat(save_path,"/PACF.csv"));
    end
end
