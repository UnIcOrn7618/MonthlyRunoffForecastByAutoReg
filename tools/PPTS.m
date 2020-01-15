function [ppts] = PPTS(y_true,y_pred,gamma)
[y_true_,y_true_order]=sort(y_true,'descend')
y_pred_ = y_pred(y_true_order,:)
N = length(y_true);
G = round(N*gamma/100);
y = y_true_(1:G,:);
yp= y_pred_(1:G,:);
abs_v = abs((y-yp)./y*100);
sum_abs = sum(abs_v);
ppts = sum_abs*100/(gamma*N);
end

