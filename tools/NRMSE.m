function [nrmse] = NRMSE(y_true,y_pred)
N = length(y_true);
nrmse = (sqrt(sum((y_true-y_pred).^2)/N))/(sum(y_true)/N);
end

