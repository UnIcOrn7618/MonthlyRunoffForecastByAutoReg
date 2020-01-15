clear
close all
clc

addpath('./EEMD');

data_path='../time_series/';

station='Zhangjiashan' %'Huaxian', 'Xianyang' or 'Zhangjiashan'

switch station
    case 'Huaxian'
        save_path = '../Huaxian_eemd/data/';
        data = xlsread([data_path,'HuaxianRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
        
    case 'Xianyang'
        save_path = '../Xianyang_eemd/data/';
        data = xlsread([data_path,'XianyangRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
        
    case 'Zhangjiashan'
        save_path = '../Zhangjiashan_eemd/data/';
        data = xlsread([data_path,'ZhangJiaShanRunoff1953-2018(1953-2018).xlsx'],1,'B2:B793');%full=792samples
end

if exist([save_path,'eemd-test'])==0
    mkdir([save_path,'eemd-test']);
end
    
    
    
%Decompose the entire set
signals = eemd(data,0.2,100);
[m,n] = size(signals);
columns = {};
for i=1:n
    if i==1
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i-1)];
    end
end
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,'EEMD_FULL.csv']);
    
% Decompose the training set
train_len = 552;
train=data(1:train_len);%train
train_signals = eemd(train,0.2,100);
train_decompositions = array2table(train_signals, 'VariableNames', columns);
writetable(train_decompositions, [save_path,'EEMD_TRAIN.csv']);

% Decompose the training-development sey
train_dev_len = 672;
train_dev = data(1:train_dev_len);
train_dev_signals = eemd(train_dev,0.2,100);
train_dev_decompositions = array2table(train_dev_signals, 'VariableNames', columns);
writetable(train_dev_decompositions, [save_path,'EEMD_TRAINDEV.csv']);
save(['../results_analysis/results/',station,'-eemd.mat']);
    
    
% Decompose the appended set
parfor i=1:240%1:240
    test_num=i;
    appended_signals = eemd(data(1:(train_len+test_num)),0.2,100);%dev2-test
    appended_decompositions = array2table(appended_signals, 'VariableNames', columns);
    a2=[save_path,'eemd-test/eemd_appended_test'];
    b2=num2str(train_len+test_num);
    c2='.csv';
    abc2=[a2,b2,c2];
    writetable(appended_decompositions, abc2)
end
    
% [m,n] = size(allmodels);
% t=1:m;
% t=t';
% raw = allmodels(:,1);
% parfor i=2:n
%     if i==n
%         eval(['R',num2str(i-1),'=','allmodels(:,i)',';']);
%     else
%         eval(['IMF',num2str(i-1),'=','allmodels(:,i)',';']);
%     end
% end
    
    
    