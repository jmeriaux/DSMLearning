######################################
# Practical machine learning - project
#######################################


#Common libraries
library(caret);

#Install doMC package for multi-core processing
library(doMC)
registerDoMC(cores = 4);

#Utility to convert factors in numeric
asNumeric <- function(x) as.numeric(as.character(x))
factorsNumeric <- function(d) modifyList(d, lapply(d[, sapply(d, is.factor)],   

#Select numeric variables after visual inspection
listcol <- c(8:11,37:49,60:68,84:85,102,113:124,140,151);

#Functions to return results
answers = rep("A", 20);

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}



######################## Data load ####################
#Whole training set
trset <- read.csv("./data/pml-training.csv",header=TRUE);

#Final test
tstset <- read.csv("./data/pml-testing.csv",header=TRUE);

#Create training and test set 
trainIndex <-  createDataPartition(y=trset$classe, p = 0.75,list=FALSE)
training = trset[trainIndex,]
testing = trset[-trainIndex,]


￼# print nearZeroVar table
t <- nearZeroVar(training,saveMetrics=TRUE)
t <- t[t$nzv == TRUE,];
#check that selected column are not in nearZeroVar
colnames(straining) %in% rownames(t)

######################################
#Prepare subset of the data
i <- listcol;
straining <-  factorsNumeric(training[,i]);
stesting <- factorsNumeric(testing[,i]);
ftest <- factorsNumeric(tstset[,i]);

ncol <- ncol(straining);

#########################################
#Preprocessing using scaling and centering
ilpreproc <- preProcess(straining,method=c("center","scale"), verbose=TRUE);
trainPreproc <- predict(ilpreproc,straining);
testPreproc <- predict(ilpreproc,stesting);
ftestPreproc <- predict(ilpreproc,ftest);

###########################################
#Append class to training set
trainl <- cbind(trainPreproc,training$classe);
colnames(trainl)[[ncol(trainl)]] <- "classe";


###########Train models using training set 

######## Stochastic Gradient Boosting  - boosted tree model
modfitgbm <- train(classe ~ ., data=trainl, method="gbm" );

#Predict on test data - gbm
resgbm <- predict(modfitgbm,newdata=testPreproc);
confgbm <- confusionMatrix(testing$classe,resgbm);


####### Stochastic Gradient Boosting with cross validation
# Training control (optional)  - 10 times  and 10 fold cross validation
# Reference http://topepo.github.io/caret/training.html
fitControl <- trainControl(method = "repeatedcv", number = 10,  repeats = 10);
modfitgbmcv <- train(classe ~ ., data=trainl, method="gbm" ,trControl=fitControl );

#Predict on test data - gbm
resgbmcv <- predict(modfitgbmcv,newdata=testPreproc);
confgbmcv <- confusionMatrix(testing$classe,resgbmcv);


####### Random forest ############
modfitrf <- train(classe ~ .,data=trainl,method="rf",prox=TRUE );

#Predict on test data - rf
resbrf <- predict(modfitrf,newdata=testPreproc);
confbrf <- confusionMatrix(testing$classe,resbrf);


##### Write result from seleted model
#Write results for test
fres <- predict(modfitrf,newdata=ftestPreproc);

answers <- fres;
pml_write_files(answers)

save.image("MLPRJ");

