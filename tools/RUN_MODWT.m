% !Thanks John Quilty for providing this MATLAB script
% MATLAB code to calculate boundary-corrected wavelet coefficients 
% using MODWT (should work in MATLAB R2016a and above)
clear
close all
clc
data_path='../time_series/';
station='Huaxian' %'Huaxian', 'Xianyang' or 'Zhangjiashan'
switch station
    case 'Huaxian'
        save_path = '../Huaxian_modwt/data/';
        data = xlsread([data_path,'HuaxianRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
    
    case 'Xianyang'
        save_path = '../Xianyang_modwt/data/';
        data = xlsread([data_path,'XianyangRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
        
    case 'Zhangjiashan'
        save_path = '../Zhangjiashan_modwt/data/';
        data = xlsread([data_path,'ZhangJiaShanRunoff1953-2018(1953-2018).xlsx'],1,'B2:B793');%full=792samples
end

% % set decomposition parameters
wname = 'db10'; % wavelet filter (change to suit needs)
[g, h] = wfilters(wname,'r'); % get reconstruction low ('g') and high ('h') pass filters
L = numel(g); % number of filter coefficients
J = 2; % decomposition level (change to suit needs)
L_J = (2^J - 1)*(L - 1) + 1; % number of boundary-coefficients at beginning of time series (remove these)

columns = {};
for i=1:J+2%±íÍ·
    if i==J+1
       columns{i}=['A',num2str(J)];
    elseif i==J+2
        columns{i}='ORIG';
    else
        columns{i}=['D',num2str(i)];
    end
end
if exist([save_path,wname,'-lev',num2str(J)],'dir')==0
   mkdir([save_path,wname,'-lev',num2str(J)]);
end
if exist([save_path,wname,'-lev',num2str(J),'/modwt-test/'],'dir')==0
   mkdir([save_path,wname,'-lev',num2str(J),'/modwt-test/']);
end

% Decompose the entire set
X=data;
coefs = modwt(X, wname, J); % get second level MODWT wavelet and scaling coefficients
W_bc = coefs(1:J,L_J+1:end).'; % boundary-corrected wavelet coefficients
V_bc = coefs(J+1,L_J+1:end).'; % boundary-corrected scaling coefficients
signals=[W_bc,V_bc,X(L_J+1:end)];
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,wname,'-lev',num2str(J),'/MODWT_FULL.csv'])

% Decompose the training set
X=data(1:552);
coefs = modwt(X, wname, J); % get second level MODWT wavelet and scaling coefficients
W_bc = coefs(1:J,L_J+1:end).'; % boundary-corrected wavelet coefficients
V_bc = coefs(J+1,L_J+1:end).'; % boundary-corrected scaling coefficients
signals=[W_bc,V_bc,X(L_J+1:end)];
decompositions = array2table(signals, 'VariableNames', columns);
writetable(decompositions, [save_path,wname,'-lev',num2str(J),'/MODWT_TRAIN.csv'])

% Decompose the appedend set
for i=1:240
    %the test set
    X=data(1:(552+i));
    coefs = modwt(X, wname, J); % get second level MODWT wavelet and scaling coefficients
    W_bc = coefs(1:J,L_J+1:end).'; % boundary-corrected wavelet coefficients
    V_bc = coefs(J+1,L_J+1:end).'; % boundary-corrected scaling coefficients
    signals=[W_bc,V_bc,X(L_J+1:end)];
    decompositions = array2table(signals, 'VariableNames', columns);
    a2=[save_path,wname,'-lev',num2str(J),'/modwt-test/modwt_appended_test'];
    b2=num2str(552+i);
    c2='.csv';
    abc2=[a2,b2,c2];
    writetable(decompositions, abc2)
end
