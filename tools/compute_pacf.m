close all;
clear
clc
%%================================vmd=========================================================================
% load('huaxian_vmdtrain.mat')
% load('xianyang_vmdtrain.mat')
% load('zhangjiashan_vmdtrain.mat')
% NumLags=20
% NumImfs=size(imf,2)
% pacfs=zeros(NumLags+1,NumImfs);
% up_bounds=zeros(NumLags+1,NumImfs);
% lo_bounds=zeros(NumLags+1,NumImfs);
% for i=1:NumImfs
%     [pacf,lags,bounds] = parcorr(imf(:,i),'NumLags',NumLags);
%     pacfs(:,i)=pacf;
%     up_bounds(:,i)=bounds(1);
%     lo_bounds(:,i)=bounds(2);
% end
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-vmd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-vmd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-vmd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-vmd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-vmd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-vmd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);
 
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-vmd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-vmd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-vmd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

%%==================================eemd=============================================================================
% load('huaxian_eemdtrain.mat')
% load('xianyang_eemdtrain.mat')
% load('zhangjiashan_eemdtrain.mat')
% NumLags=20
% NumImfs=size(allmodels,2)-1
% pacfs=zeros(NumLags+1,NumImfs);
% up_bounds=zeros(NumLags+1,NumImfs);
% lo_bounds=zeros(NumLags+1,NumImfs);
% for i=1:NumImfs
%     [pacf,lags,bounds] = parcorr(allmodels(:,i+1));
%     pacfs(:,i)=pacf;
%     up_bounds(:,i)=bounds(1);
%     lo_bounds(:,i)=bounds(2);
% end
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-eemd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-eemd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-eemd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-eemd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-eemd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-eemd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);
 
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-eemd\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-eemd\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-eemd\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

%%==========================================WD(DWT)=============================================================
% load('huaxian_wdtrain.mat')
% load('xianyang_wdtrain.mat')
% load('zhangjiashan_wdtrain.mat')
% NumLags=20
% NumImfs=size(signals,2)-1
% pacfs=zeros(NumLags+1,NumImfs);
% up_bounds=zeros(NumLags+1,NumImfs);
% lo_bounds=zeros(NumLags+1,NumImfs);
% for i=1:NumImfs
%     [pacf,lags,bounds] = parcorr(signals(:,i));
%     pacfs(:,i)=pacf;
%     up_bounds(:,i)=bounds(1);
%     lo_bounds(:,i)=bounds(2);
% end
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-wd\data\db10-lev2\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-wd\data\db10-lev2\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian-wd\data\db10-lev2\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-wd\data\db10-lev2\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-wd\data\db10-lev2\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang-wd\data\db10-lev2\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);
 
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-wd\data\db10-lev2\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-wd\data\db10-lev2\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan-wd\data\db10-lev2\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

%%==========================================ssa=============================================================
% load('huaxian_ssatrain.mat')
% load('xianyang_ssatrain.mat')
load('zhangjiashan_ssatrain.mat')
NumLags=20
NumImfs=size(signals,2)
pacfs=zeros(NumLags+1,NumImfs);
up_bounds=zeros(NumLags+1,NumImfs);
lo_bounds=zeros(NumLags+1,NumImfs);
for i=1:NumImfs
    [pacf,lags,bounds] = parcorr(signals(:,i));
    pacfs(:,i)=pacf;
    up_bounds(:,i)=bounds(1);
    lo_bounds(:,i)=bounds(2);
end
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian_ssa\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian_ssa\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Huaxian_ssa\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);

% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang_ssa\data\pacfs',num2str(NumLags),'.csv'],pacfs);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang_ssa\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
% csvwrite(['H:\MonthRunoffForecastByAutoReg\Xianyang_ssa\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);
 
csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan_ssa\data\pacfs',num2str(NumLags),'.csv'],pacfs);
csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan_ssa\data\up_bounds',num2str(NumLags),'.csv'],up_bounds);
csvwrite(['H:\MonthRunoffForecastByAutoReg\Zhangjiashan_ssa\data\lo_bounds',num2str(NumLags),'.csv'],lo_bounds);
