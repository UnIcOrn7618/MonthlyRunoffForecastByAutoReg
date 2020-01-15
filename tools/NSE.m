function [nse] = NSE(y_true,y_pred)
nse = 1-sum((y_true-y_pred).^2)/sum((y_true-mean(y_true)).^2);
end

