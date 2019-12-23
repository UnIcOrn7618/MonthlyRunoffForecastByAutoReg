clear
close all
clc

%%huaxian
% data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
% save_path = 'F:\MonthRunoffForecastByAutoReg\Huaxian_eemd\data\';
% f = xlsread([data_path,'HuaxianRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%xianyang
% data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
% save_path = 'F:\MonthRunoffForecastByAutoReg\Xianyang_eemd\data\';
% f = xlsread([data_path,'XianyangRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%zhangjiashan
data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
save_path = 'F:\MonthRunoffForecastByAutoReg\Zhangjiashan_eemd\data\';
f = xlsread([data_path,'ZhangJiaShanRunoff1953-2018(1953-2018).xlsx'],1,'B2:B793');%full=792samples

% data=f;%full
training_len = 672
data=f(1:training_len);%train

allmodels = eemd(data,0.2,100);
[m,n] = size(allmodels);
columns = {};
for i=1:n
    if i==1
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i-1)];
    end
end
decompositions = array2table(allmodels, 'VariableNames', columns);
writetable(decompositions, [save_path,'EEMD_TRAIN672.csv'])
% writetable(decompositions, [save_path,'EEMD_FULL.csv'])



% for i=1:240%1:240
%     test_num=i;
%     allmodels = eemd(f(1:(552+test_num)),0.2,100);%dev2-test
%     [m,n] = size(allmodels);
%     columns = {};
%     for i=1:n
%         if i==1
%             columns{i}='ORIG';
%         else
%             columns{i}=['IMF',num2str(i-1)];
%         end
%     end
%     decompositions = array2table(allmodels, 'VariableNames', columns);
%     a2=[save_path,'eemd-test/eemd_appended_test'];
%     b2=num2str(552+test_num);
%     c2='.csv';
%     abc2=[a2,b2,c2];
%     writetable(decompositions, abc2)
% end

% [m,n] = size(allmodels);
% t=1:m;
% t=t';
% raw = allmodels(:,1);
% for i=2:n
%     if i==n
%         eval(['R',num2str(i-1),'=','allmodels(:,i)',';']);
%     else
%         eval(['IMF',num2str(i-1),'=','allmodels(:,i)',';']);
%     end
% end


    