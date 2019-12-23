clear;
close all;
clc;
data_path='F:\MonthRunoffForecastByAutoReg\time_series\';
station='Zhangjiashan'

% Time Domain 0 to T     时域  0到T
%%huaxian

save_path = 'F:\MonthRunoffForecastByAutoReg\Huaxian_vmd\data\';
data = xlsread([data_path,'HuaxianRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%xianyang

% save_path = 'F:\MonthRunoffForecastByAutoReg\Xianyang_vmd\data\';
% data = xlsread([data_path,'XianyangRunoff1951-2018(1953-2018).xlsx'],1,'B26:B817');%full=792samples
%%zhangjiashan

% save_path = 'F:\MonthRunoffForecastByAutoReg\Zhangjiashan_vmd\data\';
% data = xlsread([data_path,'ZhangJiaShanRunoff1953-2018(1953-2018).xlsx'],1,'B2:B793');%full=792samples


% f = data;%full
training_len = 672
f = data(1:training_len)%train

orig=f;
T = length(f);
fs = 1/T;

f=f';
t = (1:T)/T;
freqs = t-0.5-1/T;

% % some sample parameters for VMD             VMD 的一些参数设置
alpha = 2000;        % moderate bandwidth constraint        宽带限制
tau = 0;            % noise-tolerance (no strict fidelity enforcement)   噪声容忍度
K = 7;              % 4 modes    
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly    参数初始化
tol = 1e-9;

columns = {};
for i=1:(K+1)
    if i==(K+1)
        columns{i}='ORIG';
    else
        columns{i}=['IMF',num2str(i)];
    end
end

%%--------------- Run actual VMD code   运行VMD 指令
[imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);%u为13*10958的矩阵
figure
plot(omega); 
imf = imf';

allmodels=[imf,orig];
decompositions = array2table(allmodels, 'VariableNames', columns);
% file_name=['VMD_TRAIN_K',num2str(K),'_a',num2str(alpha),'.csv'];
file_name=['VMD_TRAIN.csv'];
% writetable(decompositions, [save_path,file_name])


figure
subplot(211)
plot(t,f,'b')
set(gca,'FontSize',8,'XLim',[0 t(end)]);
title('Original signal')
xlabel('Number of Time(day)')
ylabel('Daily inflow(m3)');
subplot(212)
% [Yf, f] = FFTAnalysis(x, Ts);
plot(freqs,abs(fft(f)),'b')
title('The spectrum of the original signal')
xlabel('f/Hz')
ylabel('|Y(f)|');

for k1 = 0:4:K-1
    figure
    for k2 = 1:min(4,K-k1)
        subplot(4,2,2*k2-1)
        plot(t,imf(:,k1+k2),'b')
        set(gca,'FontSize',8,'XLim',[0 t(end)]);
        title(sprintf('IMF%d', k1+k2))
        xlabel('Time/s')
        ylabel(sprintf('IMF%d', k1+k2));
        
        subplot(4,2,2*k2)
%         [yf, f] = FFTAnalysis(imf(k1+k2,:), fs);        
        plot(freqs, abs(fft(imf(:,k1+k2))),'b')
        title(sprintf('The spectrum of IMF%d', k1+k2))
        xlabel('f/Hz')
        ylabel(sprintf('|IMF%d(f)|',k1+k2));
    end
end

figure
subplot(4,2,2*k2-1)
plot(t,imf(:,k1+k2),'b')
set(gca,'FontSize',8,'XLim',[0 t(end)]);
title(sprintf('IMF%d', k1+k2))
xlabel('Time/s')
ylabel(sprintf('IMF%d', k1+k2));
        
subplot(4,2,2*k2)
%[yf, f] = FFTAnalysis(imf(k1+k2,:), fs);        
plot(freqs, abs(fft(imf(:,k1+k2))),'b')
title(sprintf('The spectrum of IMF%d', k1+k2))
xlabel('f/Hz')
ylabel(sprintf('|IMF%d(f)|',k1+k2));
savefig([station,'_vmd_k',num2str(K),'_a',num2str(alpha),'.fig']);

% for i=1:240  %1:240
%     test_num=i;
%     f = data(1:(552+test_num));
%     orig=f;
%     f=f';
%     % run vmd
%     [imf, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
%     imf = imf';
%     %save vmd
%     allmodels=[imf,orig];
%     decompositions = array2table(allmodels, 'VariableNames', columns);
%     a2=[save_path,'vmd-test/vmd_appended_test'];
%     b2=num2str(552+test_num);
%     c2='.csv';
%     abc2=[a2,b2,c2];
%     writetable(decompositions, abc2)
% end
