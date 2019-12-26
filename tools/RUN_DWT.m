clear
close all
clc
%%huaxian
% data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
% save_path = 'F:\MonthRunoffForecastByAutoReg\Huaxian_wd\data\';
% f = xlsread([data_path,'HuaxianRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%xianyang
% data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
% save_path = 'F:\MonthRunoffForecastByAutoReg\Xianyang_wd\data\';
% f = xlsread([data_path,'XianyangRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%zhangjiashan
data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
save_path = 'F:\MonthRunoffForecastByAutoReg\Zhangjiashan_wd\data\';
f = xlsread([data_path,'ZhangJiaShanRunoff1953-2018(1953-2018).xlsx'],1,'B2:B793');%full=792samples
%% wavelet decomposition======start
%%%%%%%% set the hyperparameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% level=round(log10(l_td));%log10(552)=2.741939077729199
lev = 2 %same decomposition level as VMD
columns = {};
for i=1:lev+2%±íÍ·
    if i==lev+1
       columns{i}=['A',num2str(lev)];
    elseif i==lev+2
        columns{i}='ORIG';
    else
        columns{i}=['D',num2str(i)];
    end
end
mother_wavelet = 'db10',%'db10'
if exist([save_path,mother_wavelet,'-lev',num2str(lev)],'dir')==0
   mkdir([save_path,mother_wavelet,'-lev',num2str(lev)]);
end
if exist([save_path,mother_wavelet,'-lev',num2str(lev),'/wd-test/'],'dir')==0
   mkdir([save_path,mother_wavelet,'-lev',num2str(lev),'/wd-test/']);
end

% data=f;%full
training_len = 672
data=f(1:training_len);%train

len=length(data);%the length of data
%%% Performe decomposition of the data set
[C,L]=wavedec(data,lev,mother_wavelet);
%%% Extract approximation and detail coefficients
%%% Extract the approximation coefficients from C
cA=appcoef(C,L,mother_wavelet,lev);
%%% Extract the detail coefficients from C
cD = detcoef(C,L,linspace(1,lev,lev));
%%% Reconstruct the level approximation and level details
A=wrcoef('a',C,L,mother_wavelet,lev); %the approximation
for i=1:lev
    eval(['D',num2str(i),'=','wrcoef(''d'',C,L,mother_wavelet,i)',';']); %the details
end
%%% combine the details, appromaximation and original data into a single parameter
signals=zeros(len,lev+2);
signals(:,lev+1)=A;
signals(:,lev+2)=data;
for i=1:lev
    eval(['signals(:,i)','=','D',num2str(i),';']);
end
%%% save the decomposition results and the original data
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,mother_wavelet,'-lev',num2str(lev),'/WD_TRAIN672.csv'])
% 
% %%plot the decomposition results and original data
% f1=figure
% set(f1,'position',[1500 1500 900 900]);
% %%% tight_subplot(rowas,columns,[v-space,h-space],[bottom,top],[left,right])
% ha = tight_subplot(lev+2,1,[.05 .08],[.05 .04],[.05 .02])
% for i=1:size(signals,2)
%     
%     if i==1
%         axes(ha(i));
%         plot(data,'r');
%         title('Original set');
%     elseif i==2
%         axes(ha(i));
%         plot(A,'b');
%         title(['Approximation A',num2str(lev)]);
%     else 
%         axes(ha(i));
%         plot(signals(:,i-2),'g');
%         title(['Detail D',num2str(i-2)]);
%     end
% end
%% wavelet decomposition for train-dev set======end

%%%%%%%%%%%%% decompose the test set one by one %%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:240
%     %the test set
%     test_num=i;
%     test=f(1:(552+test_num));
%     %performe decomposition
%     [C,L]=wavedec(test,lev,mother_wavelet);
%     % Extract approximation and detail coefficients
%     % Extract the level 3 approximation coefficients from C
%     cA=appcoef(C,L,mother_wavelet,lev);
%     % Extract the level 3,2,and 1 detail coefficients from C
%     cD = detcoef(C,L,linspace(1,lev,lev));
%     % Reconstruct the level 3 approximation and level 1,2,3  details
%     A=wrcoef('a',C,L,mother_wavelet,lev);
%     for j=1:lev
%         eval(['D',num2str(j),'=','wrcoef(''d'',C,L,mother_wavelet,j)',';']); %the details
%     end
%     %combine the details, approximation and orig into one variable
%     signals=zeros(length(test),lev+2);
%     signals(:,lev+1)=A;
%     signals(:,lev+2)=test;
%     for j=1:lev
%         eval(['signals(:,j)','=','D',num2str(j),';']);
%     end
%     % save the decomposition results and the original data
%     decompositions = array2table(signals, 'VariableNames', columns);
%     a2=[save_path,mother_wavelet,'-lev',num2str(lev),'/wd-test/wd_appended_test'];
%     b2=num2str(552+test_num);
%     c2='.csv';
%     abc2=[a2,b2,c2];
%     writetable(decompositions, abc2)
% end

