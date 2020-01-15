library('ggplot2')
library('forecast')
library('tseries')
library("readxl")

MSE<-function(y_pred,y_true){
    return (mean((as.numeric(unlist(y_pred))-as.numeric(unlist(y_true))^2)))
}

eval_arima_model<-function(train,dev,order){
    model=arima(train,order = order)
    dev_pred = predict(model,120)
    error = MSE(y_pred=dev_pred,y_true=dev)
    return (error)
}

eval_arima_models<-function(train,dev,p_values,d_values,q_values){
    best_score = Inf
    best_order = NULL
    for(i in 1:length(p_values)){
        for(j in 1:length(d_values)){
            for(k in 1:length(q_values)){
                order=c(p_values[i],d_values[j],q_values[k])
                print(order)
                mse = eval_arima_model(train,dev,order = order)
                if(mse < best_score){
                    best_score=mse
                    best_order=order
                }
            }
        }
    }
    print(best_order)
    print(best_score)
    return (best_order)
}
huaxian=read_excel("time_series/HuaxianRunoff1951-2018(1953-2018).xlsx")
hua_mr = huaxian$MonthlyRunoff[25:816]
hua_ts = ts(hua_mr,frequency = 12)
decomp = stl(hua_ts,s.window = "periodic")
deseasonal_hua <- seasadj(decomp)

auto.arima(deseasonal_hua,seasonal=FALSE)

train = deseasonal_hua[1:552]
dev = deseasonal_hua[553:672]
test = deseasonal_hua[673:792]
# length(train)
# length(dev)
# length(test)
p_values=c(1,2,3,4,5,6,7,8,9)
d_values=c(1,2,3)
q_values=c(1,2,3)
best_order = eval_arima_models(train,dev,p_values,d_values,q_values)
model = arima(train,order = best_order)
dev_pred = predict(model,120)
print(dev_pred)
plot(as.numeric(unlist(dev_pred)))
