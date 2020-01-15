function [c] = plot_pred(y_true,y_pred,file_name)
figure('Name','Predictions and correlations','position',[200 0 1436 576])
% tight_subplot(rowas,columns,[v-space,h-space],[bottom,top],[left,right])
ha = tight_subplot(1,2,[.03 .04],[.1 .03],[.04 .02])
lw =1.3;
axes(ha(1));
t_p=plot(y_true,'b','LineWidth',lw);
label_t = 'Records';
hold on
p_p=plot(y_pred,'r--','LineWidth',lw);
label_p = 'Predictions';
legend([t_p;p_p],label_t,label_p,'Location','northwest');
hold off
xlabel('Time(month)');
ylabel('Flow(10^8m^3)');
xlim([1,length(y_true)]);
axes(ha(2));
axis('equal');
xymin=min(y_true);
xymax=max(y_true);
if xymin>min(y_pred)
    xymin = min(y_pred);
elseif xymax<max(y_pred)
    xymax = max(y_pred);
end
x = linspace(xymin,xymax)
c = polyfit(y_pred,y_true,1)
y = x*c(1)+c(2)
disp(['y = ' num2str(c(1)) '*x + ' num2str(c(2))])
y_est = polyval(c,x);
scatter(y_pred,y_true,'b');
hold on
ideal = plot([xymin,xymax],[xymin,xymax],'k--','LineWidth',lw);
label_ideal = 'Ideal fit';
linear = plot(x,y,'r--','LineWidth',lw);
label_linear = 'Linear fit';
legend([ideal;linear],label_ideal,label_linear,'Location','northwest');
xlim([xymin,xymax]);
ylim([xymin,xymax]);
xlabel('Predictions(10^8m^3)');
ylabel('Records(10^8m^3)');
saveas(gcf,file_name);
end

