#Prediction Models Functions
#Libraries-------------
library(caret)
library(ggplot2)
library(randomForest)
library(rminer)
library(gbm)
library(pls)
library(glmnet)
library(party)
library(kernlab)

#Functions in R script were modified by Valerie Odeh-Couvertier versions

#Random Forest---------------
RFfunct <- function(data,test,y,x,meval='all',ntree=c(500,1000,1500,2000,2500), metric="Rsquared"){
  #Data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  #Custom function to tune both RF parameters in a grid
  customRF <- list(type = "Regression", library = "randomForest", loop = NULL)
  customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
  customRF$grid <- function(x, y, len = NULL, search = "grid") {}
  customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
    randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
  }
  customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
    predict(modelFit, newdata)
  customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
    predict(modelFit, newdata, type = "prob")
  customRF$sort <- function(x) x[order(x[,1]),]
  customRF$levels <- function(x) x$classes
  
  #How many mtry values to try
  m=length(x)
  if(meval=='all'){
    mtry=c(1:m)
  }else{
    mtry=sqrt(m)
  }
  
  #Grid of parameters to evaluate
  tunegrid <- expand.grid(.mtry=mtry, .ntree=ntree)
  
  #Seed assignment to use in control
  size=nrow(tunegrid) #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)

  #Train & Tune
  tune <- train(Y~.,data = Data, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control,importance=TRUE)
  final <- tune$finalModel
  
  tune_p=plot(tune,main='Parameter Tuning: RF')
  #LOO Performance Metric
  pred <- final$predicted
  r2 <- mmetric(Y,pred ,metric=c("R2"))
  
  #Variable Importance List
  imp <- as.data.frame(importance(final,type=1))
  imp=cbind(row.names(imp),imp)
  imetric=names(imp)[2]
  names(imp)=c("variables","importance")
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("RF:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(final,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test, "tune_plot"=tune_p)
  
  return(rList)
}

#Gradient Boosting Trees-------------------
GBMfunct <- function(data,test, y, x,interaction.depth=c(1:4),n.trees = (1:20)*10, shrinkage=c(0.1,0.01, 0.02),n.minobsinnode=c(2:6),bag.fraction=0.5,metric="Rsquared"){
  #data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  # grid of GBM parameter combinations to evaluate
  tunegrid=expand.grid(interaction.depth=interaction.depth, n.trees = n.trees, shrinkage=shrinkage,n.minobsinnode=n.minobsinnode)
  
  #Seed assignment to use in control
  size=nrow(tunegrid) #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)
  
  #Tuning
  tune <- train(Y~.,data=Data, method="gbm", distribution="gaussian", tuneGrid=tunegrid, metric=metric, trControl=control,bag.fraction=bag.fraction,verbose=FALSE)
  final <- tune$finalModel
  
  tune_p=plot(tune,main='Parameter Tuning: GBM')
  
  #LOO Performance Metric
  pred <- final$fit
  r2 <- mmetric(Y,pred ,metric=c("R2"))
  
  #Variable Importance List
  imp <- as.data.frame(summary(final,plot=FALSE))
  imetric=names(imp)[2]
  names(imp)=c("variables","importance")
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("GBM:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(final,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test,"tune_plot"=tune_p)
  
  return(rList)
}


#PLSR-----------
PLSRfunct=function(data,test, y, x,metric="Rsquared"){
  #data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  # grid of PLSR ncomp to evaluate
  tuneLength = 15
  
  #Seed assignment to use in control
  size=tuneLength #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)
  
  #Tuning
  tune <- train(Y~.,data=Data, method="pls",preProc = c("center", "scale"),tuneLength = 15,metric=metric )
  final <- tune$finalModel
  
  tune_p=plot(tune,main='Parameter Tuning: PLSR')
  
  #LOO Performance Metric
  pred <- final$fitted.values[,,tune$bestTune$ncomp]
  r2 <- mmetric(Y,pred ,metric=c("R2"))
  
  #Variable Importance List
  imp <- as.data.frame(varImp(final,tune$bestTune$ncomp))
  imp=cbind(row.names(imp),imp)
  names(imp)=c("variables","importance")
  imetric="weighted sums of absolute coefficients"
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("PLSR:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(tune,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test,"tune_plot"=tune_p)
  
  return(rList)
}

#SVM------
SVMfunct <- function(data,test,y, x,C=seq(0.05,2,.05),metric="Rsquared"){
  #data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  # grid of SVMLinear (kernlab) parameter combinations to evaluate
  tunegrid=expand.grid(C=C)
    
  #Seed assignment to use in control
  size=nrow(tunegrid) #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)
  
  #Tuning
  tune <- train(Y~.,data=Data, method="svmLinear", preProcess = c("center","scale"),tuneGrid=tunegrid, metric=metric, trControl=control)
  final <- tune$finalModel
  
  tune_p=plot(tune,main='Parameter Tuning: SVM')
  
  #LOO Performance Metric
  allpred=as.data.frame(tune$pred)
  pred <- allpred[which(allpred$C == tune$bestTune$C),"pred"]
  r2 <- mmetric(Y,pred ,metric=c("R2"))
  
  #fit final model to get importance scores
  fsvm=rminer::fit(Y~., data=Data,model="ksvm",scale="all",C=tune$bestTune$C,kernel="vanilladot")

  #Variable Importance List
  svm.imp <- Importance(fsvm, data=Data)
  imp <- data.frame(names(Data),svm.imp$imp)
  imp=imp[-1,] #removing the Y response from the ranking
  names(imp)=c("variables","importance")
  imetric="relative importance"
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("SVM:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(final,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test,"C"=tune$bestTune$C,"epsilon"=0.1,"tune_plot"=tune_p)
  
  return(rList)
}

#Lasso------
LASSOfunct <- function(data,test,y,x,lambda=seq(0.001,0.05,by = 0.001),metric="Rsquared"){
  #data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  # grid of lasso parameter combinations to evaluate
  tunegrid=expand.grid(.alpha=1,.lambda=lambda)
  
  #Seed assignment to use in control
  size=nrow(tunegrid) #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)
  
  #Tuning
  tune <- train(Y ~ ., data=Data,method = "glmnet",trControl = control,tuneGrid=tunegrid,family="gaussian",metric="Rsquared")
  final=tune$finalModel
  lambda.min=tune$bestTune$lambda
  
  tune_p=plot(tune,main='Parameter Tuning: LASSO')
  
  #LOO Performance Metric
  allpred=tune$pred
  pred <- allpred[which(allpred$lambda == lambda.min),"pred"]
  r2 <- mmetric(Y,pred ,metric=c("R2"))

  #Variable Importance List
  imp <- as.data.frame(varImp(final,tune$bestTune$lambda))
  imp=cbind(row.names(imp),imp)
  names(imp)=c("variables","importance")
  imetric="absolute coefficient values"
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("LASSO:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(tune,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test,"tune_plot"=tune_p)
  
  return(rList)
  
}

#CIF------
CIFfunct <- function(data,test,y,x,meval='all',metric="Rsquared",ntree=100,minsplit = 6, minbucket = 3){
  #data setup
  Y <- data[,y]
  X <- data[,x]
  Data <- cbind(Y,X)
  
  #How many mtry values to try
  m=length(x)
  if(meval=='all'){
    mtry=c(1:m)
  }else{
    mtry=sqrt(m)
  }
  
  #Grid of parameters to evaluate
  tunegrid <- expand.grid(.mtry=mtry)
  
  #Seed assignment to use in control
  size=nrow(tunegrid) #number of parameter combinations to evaluate
  seeds <- vector(mode = "list", length = nrow(X)+1)
  num=1000*ceiling(size/1000)
  for(i in 1:nrow(X)) seeds[[i]] <- sample(num,size)
  # For the final model
  seeds[[nrow(X)+1]] <- 1
  
  control <- trainControl(method = "LOOCV", seed=seeds)

  # train and tune model
  tune <- train(Y~., data=Data,method='cforest',trControl=control,tuneGrid=tunegrid,metric=metric, controls = party::cforest_unbiased(ntree = ntree,minsplit = minsplit, minbucket = minbucket))
  final=tune$finalModel
  best.mtry=tune$bestTune$mtry
  
  tune_p=plot(tune,main='Parameter Tuning: CIF')
  
  #LOO Performance Metric
  allpred=tune$pred
  pred <- allpred[which(allpred$mtry == best.mtry),"pred"]
  r2 <- mmetric(Y,pred ,metric=c("R2"))
  
  #Variable Importance List
  imp <- as.data.frame(varImp(final,conditional=TRUE))
  imp=cbind(row.names(imp),imp)
  names(imp)=c("variables","importance")
  imetric="permuted coefficients"
  
  #Variable Importance Plot
  vip=ggplot(data=imp, aes(x=reorder(variables,importance), y=importance)) + labs(x="",y=paste("importance:",imetric),title = paste("CIF:",colnames(data)[y]),sep="")+
    geom_point() + coord_flip() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(), axis.line = element_line(colour = "black"))
  #To be performed with training and testing is requested
  if(is.null(test)){
    #do nothing
    r2test="NA"
  }else{
    tpred=predict(final,newdata=test[,c(x)])
    r2test=mmetric(test[,c(y)],tpred,metric=c("R2"))
  }
  
  #output
  rList <- list("model" = final,"R2" = r2,"imp"=imp,"imp_plot"= vip,"R2test"=r2test,"tune_plot"=tune_p)
  
  return(rList)
}