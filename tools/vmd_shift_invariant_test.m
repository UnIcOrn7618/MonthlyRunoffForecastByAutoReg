% Example MATLAB script, inspired by John Quilty, to show the
% shift-invariancy and sensitivity to adding additional data points when
% using VMD for to provide inputs for operational forecasting tasks.
clear;
close all;
clc;

addpath('./VMD');

sunspot = readtable('../time_series/sunspot.csv');
x=sunspot.sunspot;
year = sunspot.year;

x0=x(1:385);
x1=x(2:386);

%% some sample parameters for VMD             VMD 的一些参数设置
alpha = 2000;       % moderate bandwidth constraint        宽带限制
tau = 0;            % noise-tolerance (no strict fidelity enforcement)   噪声容忍度
K = 8;              % number of modes, i.e., decomposition level 分解个数
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly    参数初始化
tol = 1e-9;         % the convergence tolerance   收敛允许误差

% CHECK 1: is VMD shift-invariant?
% If yes, any shifted copy of an IMF from a VMD decomposition, similar to a
% shifted copy of the original time series, should be maintained.

% For example, given the sunspot time series x (of length 386) we can
% generate a 1-step advanced copy of the original time series as follows:
% x0=(1:385)
% x1=(2:386) this is a 1-step advanced version of x0
% Observiously, shift-invariancy is preserved between x0 and x1 since
% x0(2:385)=x1(1:384)

% For shift-invariancy to be preserved for VMD, we would observe, for
% example, that the VMD IMF1 components for x0 (imf1 of x0) and x1 (imf1 of
% x1) should be exact copies of one another, advanced by a single step.
% i.e., x0_imf(2:385,1) should equal x1_imf(1:384,1) if shift-invariancy
% is preserved.

% As the case for VMD shown below, we can see the x0_imf(2:385,1) basically
% equal to x1_imf(1:384,1) except for a few samples close to the begin and
% end of x0 and x1. Interestingly, we see a low level of error close to the
% begin of the time series and a high level of error close to the end of
% the time series, of high importance in operational forecasting tasks. 
% The errors along the middle range are zeros indicating VMD is
% shift-invariant.
% We argue that the error close to the boundaries are
% caused by boundary effect, which is the exact problem this study designed
% to solve.
setdemorandstream(pi);

[x0_imf, u_hat, omega] = VMD(x0', alpha, tau, K, DC, init, tol);
x0_imf = x0_imf';
[x1_imf, u_hat, omega] = VMD(x1', alpha, tau, K, DC, init, tol);
x1_imf = x1_imf';

figure('Name','Test shift-invariancy for VMD');
hold on
x0_imf1_2_385=plot(x0_imf(2:385,1),'b');
label0=['IMF1 of x0(',num2str(year(2)),'-',num2str(year(385)),')'];
x1_imf1_1_384=plot(x1_imf(1:384,1),'r');
label1=['IMF1 of x1(',num2str(year(2)),'-',num2str(year(385)),')']
legend([x0_imf1_2_385;x1_imf1_1_384],label0,label1,'Location','northwest');
hold off

err = x0_imf(2:385,1)-x1_imf(1:384,1);
figure('Name','error between x0 and x1');
scatter(linspace(1,384,384),err,'Marker','o');

% Check the level of error (as measuured by the mean square error) between
% the imf components.
mse=mean(err.^2);

% CHECK 2: The impact of appedning data points to a time series then
% performing VMD, analogous the case in operational forecasting when new
% data becomes available and an updated forecast is made using the newly
% arrived data.

% Ideally, for forecasting situations, when new data is appended to a time
% series and some preprocessing is performed, it should not have an impact
% on previous measurements of the pre-processed time series.

% For example, if IMF1_1:N represents the IMF1, which has N total
% measurements and was derived by applying VMD to x_1:N the we would expect
% that when we perform VMD when x is appended with another measurement,
% i.e., x_1:N+1, resulting in IMF1_1:N+1 that the first 1:N measurements in
% IMF1_1:N+1 are equal to IMF1_1:N. In other words, 
% IMF1_1:N+1[1:N]=IMF1_1:N[1:N].

% We see than is not the case. Appending an additional observation to the
% time series results in the updated VMD components to be entirely
% different then the original (as of yet updated) VMD components.
% Interesting, we see a high level of error at the boundaries of the time
% seriesm, of high importance in operational forecasting tasks.

% Extend x with an additional measurement
x_1_385=x(1:385);
x_1_386=x(1:386);

% Perform VMD on original and extended time series.
[x_1_385_imf, u_hat, omega] = VMD(x_1_385', alpha, tau, K, DC, init, tol);
x_1_385_imf = x_1_385_imf';
[x_1_386_imf, u_hat, omega] = VMD(x_1_386', alpha, tau, K, DC, init, tol);
x_1_386_imf = x_1_386_imf';
figure('Name','Test sensitivity of adding additional data point');
hold on
x_1_386_imf1=plot(x_1_386_imf(1:386,1),'b');
label00=['IMF1 of x\_1\_386(',num2str(year(1)),'-',num2str(year(386)),')'];
x_1_385_imf1=plot(x_1_385_imf(1:385,1),'r');
label11=['IMF1 of x\_1\_385(',num2str(year(1)),'-',num2str(year(385)),')'];
legend([x_1_386_imf1;x_1_385_imf1],label00,label11,'Location','northwest');
hold off

% Plot error (which should be a straight line of 0s if appending an 
% additional observation has on impact on VMD)
err_append = x_1_386_imf(1:385,1)-x_1_385_imf(1:385,1)
figure('Name','error between x_1_386_imf(1:385,1) and x_1_385_imf(1:385,1)');
scatter(linspace(1,385,385),err_append,'Marker','o')

% Check the level of error (as measured by the mean square error) between
% the IMF components.
mse_append = mean(err_append.^2);

% The problem gets exasperated if it is not a single time point and that is
% updated, but several.
x_1_300=x(1:300);

% Perform VMD on the original and extended time series.
[x_1_300_imf, u_hat, omega] = VMD(x_1_300', alpha, tau, K, DC, init, tol);
x_1_300_imf = x_1_300_imf';

% Plot IMF for x_1_300 and x_1_386
figure('Name','Test sensitivity of adding several additional data points');
hold on
x_1_386_imf1=plot(x_1_386_imf(1:386,1),'b');
label00=['IMF1 of x\_1\_386(',num2str(year(1)),'-',num2str(year(386)),')'];
x_1_300_imf1=plot(x_1_300_imf(1:300,1),'r');
label11=['IMF1 of x\_1\_300(',num2str(year(1)),'-',num2str(year(300)),')'];
legend([x_1_386_imf1;x_1_300_imf1],label00,label11,'Location','northwest');
hold off

% Plot error (which should be a straight line of 0s if appending several
% additional observations has no impact on VMD)
err_append_several = x_1_300_imf(1:300,1)-x_1_386_imf(1:300,1);
figure('Name','error between x_1_300_imf(1:300,1) and x_1_386_imf(1:300,1)');
scatter(linspace(1,300,300),err_append_several,'Marker','o');

% Check the level of error (as measured by the mean square error) between
% the IMF components.
mse_append_several = mean(err_append_several.^2);





