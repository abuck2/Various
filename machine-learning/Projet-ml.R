rm(list=ls())
##Setup##Load the libraries:
library(FactoMineR) #for PCA/MCA
library(factoextra)
library(MASS) #for PCA
library(caret) #For CV
library(glmnet) #for ridge regression & lasso
library(neuralnet) #for neural net
library(NeuralNetTools) 
library(nnet) #alternative package
library(RSNNS) #Alternative package
library(class)
library(pls) #partial least square and principal component regression
library(FNN) #knn regressop,
library(e1071) #SVM
#function for plotting neuralnets :
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r') ## Plot nnet
#dependances
library(scales)
library(reshape)
library(fastICA)
library(pROC) 
library(R.matlab)

## Open the datasets and set up the data
setwd("/home/alexis/Documents/stats/annee2/ml/")
data<-readMat("data_2016.mat")
class(data)
str(data)
known<-data$X1
unknown<-data$X2
heating<-data$heating.load.1
cooling<-data$cooling.load.1

## take the name of the features and put them as names of the column
featnam<-rep(NULL, 8)
for(i in 1:8){
  data$feature.names[[i]][[1]]<-gsub(" ", "", data$feature.names[[i]][[1]], fixed = TRUE)
  featnam[i]<-data$feature.names[[i]][[1]]
}
featnam
known<-as.data.frame(known)
colnames(known)<-featnam

#Variable are categorical
for(i in 1:8){
  known[,i]<-as.numeric(known[,i]) #as.ordered
}
str(known)

unknown<-as.data.frame(unknown)
colnames(unknown)<-featnam
for(i in 1:8){
  unknown[,i]<-as.numeric(unknown[,i])
}
str(unknown)

total_h<-cbind(known, heating)
total_c<-cbind(known, cooling)
total<-cbind(total_h,cooling)
## first look at the data
dim(known)
dim(unknown)
dim(heating)
dim(cooling)
str(total)

## Do we have missing values?
print("Valeurs manquantes : ")
for(i in 1:8){
  print(sum(is.na(known[,i])))
}



#How is the data?
findLinearCombos(known) 
acp<-PCA(known)
fviz_screeplot(acp, ncp=10)
acp$eig

#Linear regression
model <- caret::train(cbind(heating+cooling)~., data=total,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10)
) 
model 
lm_rmse<-getTrainPerf(model)$TrainRMSE

model <- caret::train(cbind(heating+cooling)~., data=total,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10),
               preProcess=c("center", "scale")
) 
model 
lm_rmse_s<-getTrainPerf(model)$TrainRMSE

model <- caret::train(cbind(heating+cooling)~., data=total,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10),
               preProcess=c("center", "scale", "ica")
) 
model 
lm_rmse_i<-getTrainPerf(model)$TrainRMSE

model <- caret::train(cbind(heating+cooling)~., data=total,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10),
               preProcess=c("center", "scale", "pca")
) 
model 
lm_rmse_p<-getTrainPerf(model)$TrainRMSE

model <- caret::train(heating~., data=total_h,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10),
               preProcess=c("center", "scale", "pca")
) 
model 
lm_rmse_h<-getTrainPerf(model)$TrainRMSE

model <- caret::train(cooling~., data=total_c,
               method = "lm", 
               trControl = trainControl(method = "cv", number = 10),
               preProcess=c("center", "scale", "pca")
) 
model 
lm_rmse_c<-getTrainPerf(model)$TrainRMSE

#Compares the RMSE of linear modelling
names<-as.vector(c("lm_rmse","lm_rmse_h","lm_rmse_c","lm_rmse_s","lm_rmse_i","lm_rmse_p"))
data<-as.data.frame(c(lm_rmse,lm_rmse_h,lm_rmse_c,lm_rmse_s,lm_rmse_i,lm_rmse_p))
rownames(data)<-names
data

##regression with ElasticNet
set.seed(5) 
model <- caret::train( cooling~., data=total_c,
                method = "glmnet", 
                trControl = trainControl(method = "cv", number = 10, repeats = 15),
                preProcess=c("center", "scale"),
                tuneLength = 15
) 
model 

grid <- expand.grid(alpha=c(0.4857143), lambda=c(seq(0, 0.005, 0.00005))) 
model <- caret::train( cooling~., data=total_c,
                method = "glmnet", 
                trControl = trainControl(method = "cv", number = 10, repeats = 3),
                preProcess=c("center", "scale"),
                tuneGrid = grid
) 
model 
plot(model) 
glmnet_cool_rmse<-getTrainPerf(model)$TrainRMSE

model <- caret::train( heating~., data=total_h,
                method = "glmnet", 
                trControl = trainControl(method = "cv", number = 10, repeats = 15),
                preProcess=c("center", "scale"),
                tuneLength = 15
) 
model 
glmnet_heat_rmse<-getTrainPerf(model)$TrainRMSE

grid <- expand.grid(alpha=c(0.1642857), lambda=c(seq(0.004, 0.0055, 0.00005))) 
model <- caret::train( heating~., data=total_h,
                method = "glmnet", 
                trControl = trainControl(method = "cv", number = 10, repeats = 3),
                preProcess=c("center", "scale"),
                tuneGrid = grid
) 
model 
plot(model) 
glmnet_cool_rmse<-getTrainPerf(model)$TrainRMSE

glmnet_cool_rmse/lm_rmse_c
glmnet_heat_rmse/lm_rmse_h
#Compares the RMSE of linear modelling
names<-as.vector(c("lm_rmse","lm_rmse_h","lm_rmse_c","lm_rmse_s","lm_rmse_i","lm_rmse_p", "glmnet_cool_rmse", "glmnet_heat_rmse"))
data<-as.data.frame(c(lm_rmse,lm_rmse_h,lm_rmse_c,lm_rmse_s,lm_rmse_i,lm_rmse_p, glmnet_cool_rmse, glmnet_heat_rmse))
rownames(data)<-names
data

## KNN regression
set.seed(5) 
grid <- expand.grid(k=c(5:10)) 
model <- caret::train(heating~., data=total_h, method = "knn", 
               trControl = trainControl(method = "cv", number = 15, repeats =10),
               tuneGrid = grid,
               preProc = c("center", "scale")
)
model
plot(model)
knn_rmse_h<-getTrainPerf(model)$TrainRMSE
knn_rmse_h/lm_rmse_h

set.seed(5) 
grid <- expand.grid(k=c(2:10)) 
model <- caret::train(cooling~., data=total_c, method = "knn", 
               trControl = trainControl(method = "cv", number = 10, repeats = 3),
               tuneGrid = grid,
               preProc = c("center", "scale")
)
model
plot(model)
knn_rmse_c<-getTrainPerf(model)$TrainRMSE
knn_rmse_c/lm_rmse_c

model<-knn.reg(train = known, y = heating, k= 7)
model

#Compares the RMSE of linear modelling
names<-as.vector(c("lm_rmse","lm_rmse_h","lm_rmse_c","lm_rmse_s","lm_rmse_i","lm_rmse_p", "glmnet_cool_rmse", "glmnet_heat_rmse", "knn_rmse_h", "knn_rmse_c"))
data<-as.data.frame(c(lm_rmse,lm_rmse_h,lm_rmse_c,lm_rmse_s,lm_rmse_i,lm_rmse_p, glmnet_cool_rmse, glmnet_heat_rmse, knn_rmse_h, knn_rmse_c))
rownames(data)<-names
data

#regression with MLP/perceptron

set.seed(5) 
model <- caret::train(heating~., data=total_h, method = "mlp", 
               trControl = trainControl(method = "cv", number = 10, repeats = 2),
               tuneLength = 10,
               preProc = c("center", "scale")
)
model
plot(model)
mlp_rmse_h<-getTrainPerf(model)$TrainRMSE
mlp_rmse_h/lm_rmse_h 

set.seed(5) 
model <- caret::train(cooling~., data=total_c, method = "mlp", 
               trControl = trainControl(method = "cv", number = 10, repeats = 2),
               tuneLength = 10,
               preProc = c("center", "scale")
)
model
plot(model)
mlp_rmse_c<-getTrainPerf(model)$TrainRMSE
mlp_rmse_c/lm_rmse_c

model<-mlp(x=known, y=heating, size = 10)
plot.nnet(model)


#separately : cooling and heating with several hidden layers
set.seed(15)
grid <- expand.grid(layer1=2:6, layer2 = 1:4, layer3 = 0) 
model <- caret::train(heating~., data=total_h, method = "neuralnet", 
               trControl = trainControl(method = "cv", number = 5, repeats = 1),
               tuneGrid = grid,
               rep = 1,
               stepmax = 1e+06,
               preProc = c("center", "scale")
)
plot(model)
nnet_rmse_2_h<-getTrainPerf(model)$TrainRMSE
nnet_rmse_2_h/lm_rmse
h_model<-model

heat_model<-neuralnet(heating~GlazingAreaDistribution+GlazingArea+Orientation+OverallHeight+RoofArea+WallArea+SurfaceArea+RelativeCompactness, data=total_h, hidden = c(5,4))
plot.nnet(heat_model)

grid <- expand.grid(layer1=2:6, layer2 = 1:4, layer3 = 0) 
model <- caret::train(cooling~., data=total_c, method = "neuralnet", 
               trControl = trainControl(method = "cv", number = 5, repeats = 1),
               tuneGrid = grid,
               rep = 1,
               stepmax = 1e+06,
               preProc = c("center", "scale")
)
plot(model)
nnet_rmse_2_c<-getTrainPerf(model)$TrainRMSE
nnet_rmse_2_c/lm_rmse
c_model<-model
model

cool_model<-neuralnet(cooling~GlazingAreaDistribution+GlazingArea+Orientation+OverallHeight+RoofArea+WallArea+SurfaceArea+RelativeCompactness, data=total_c, hidden = c(6,4))
plot.nnet(cool_model)


## PCR and PLS

model <- caret::train(heating~., data=total_h, method = "pcr", 
               trControl = trainControl(method = "cv", number = 10, repeats = 5),
               preProc = c("center", "scale"),
               tuneLength=15
)
model
plot(model)
pcr_rmse_h<-getTrainPerf(model)$TrainRMSE
pcr_rmse_h/lm_rmse_h 

model <- caret::train(cooling~., data=total_c, method = "pcr", 
               trControl = trainControl(method = "cv", number = 10, repeats = 5),
               preProc = c("center", "scale"),
               tuneLength=15
)
model
plot(model)
pcr_rmse_c<-getTrainPerf(model)$TrainRMSE
pcr_rmse_c/lm_rmse_c 

#pls

model <- caret::train(heating~., data=total_h, method = "pls", 
               trControl = trainControl(method = "cv", number = 10, repeats = 5),
               preProc = c("center", "scale"),
               tuneLength=15
)
model
plot(model)
pls_rmse_h<-getTrainPerf(model)$TrainRMSE 
pls_rmse_h/lm_rmse_h 

model <- caret::train(cooling~., data=total_c, method = "pls", 
               trControl = trainControl(method = "cv", number = 10, repeats = 5),
               preProc = c("center", "scale"),
               tuneLength=15
)
model
plot(model)
pls_rmse_c<-getTrainPerf(model)$TrainRMSE 
pls_rmse_c/lm_rmse_c 


## SVR
set.seed(15)
tuneResult <- tune(svm, heating~., data=total_h,
                   ranges = list(gamma = 10^(-5:1), cost = seq(25, 525, 50))
)
print(tuneResult)

set.seed(15)
tuneResult <- tune(svm, cooling~., data=total_c,
                   ranges = list(gamma = 10^(-5:1), cost = seq(25, 275, 50))
)
print(tuneResult)

#"Zoom"

set.seed(15)
tuneResult <- tune(svm, cooling~., data=total_c,
                   ranges = list(gamma = seq(0.07,0.10,0.01), cost = seq(160, 230, 10))
)
print(tuneResult)
plot(tuneResult)

set.seed(15)
tuneResult <- tune(svm, heating~., data=total_h,
                   ranges = list(gamma = seq(0.025,0.055,0.01), cost = seq(1500, 2000, 20))
)
print(tuneResult)
plot(tuneResult)


## prédictions finales :

model <- svm(heating ~ . ,data=total_h, epsilon = 0.035 , cost=1860)
heat<-predict(model, unknown)
model <- svm(cooling ~ . ,data=total_c, epsilon = 0.08 , cost=200)
cool<-predict(model, unknown)
qqplot(cool, cooling)
qqplot(heat, heating)
writeMat( con="A.mat", Y2_1=heat, Y2_2=cool)


heat_model<-neuralnet(heating~RelativeCompactness+SurfaceArea+WallArea+RoofArea+OverallHeight+Orientation+GlazingArea+GlazingAreaDistribution, data=known, hidden = c(5,4), stepmax=1e+09)
predict(h_model, unknown)
