% !Thanks John Quilty for providing this MATLAB script
% MATLAB code to calculate boundary-corrected wavelet coefficients using MODWT (should work in MATLAB R2016a and above)
X=rand(1000,1); % random time series
wname = 'db2'; % wavelet filter (change to suit needs)
[g, h] = wfilters(wname,'r'); % get reconstruction low ('g') and high ('h') pass filters
L = numel(g); % number of filter coefficients
J = 2; % decomposition level (change to suit needs)
L_J = (2^J - 1)*(L - 1) + 1; % number of boundary-coefficients at beginning of time series (remove these)
coefs = modwt(X, wname, J); % get second level MODWT wavelet and scaling coefficients
W_bc = coefs(1:J,L_J+1:end).'; % boundary-corrected wavelet coefficients
V_bc = coefs(J+1,L_J+1:end).'; % boundary-corrected scaling coefficients