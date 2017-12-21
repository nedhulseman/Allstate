library(readr)
library(rpart)
library(lattice)
library(Matrix)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(PCAmixdata)
library(hydroGOF)
library(reshape)
library(onehot)
library(dplyr)

######################
# @Ned Hulseman      #
# @James Noe         #
# @11/12/2017        #
######################

#Reads in pre-specified training and test datasets
allstate_file<-'C:\\Users\\nedhu\\Desktop\\Fall III\\ML\\allstate_train.csv'
allstate<-read_csv(allstate_file)
final_prediction_file<-'C:\\Users\\nedhu\\Desktop\\Fall III\\ML\\allstate_test.csv'
final_prediction<-read_csv(final_prediction_file)



names(allstate)
dim(allstate) #2926 obs 132 variables
#######################
# 2926 obs            #
# 132 variables       #
# 116 categorical     #
# 14 continuous       #
# @loss is dependent  #
#######################

############################################################################################################
########################################loss exploration ###################################################
hist(allstate$loss, breaks=60)
hist(log(allstate$loss), breaks=60)
#We should probably model the log
############################################################################################################

##########################################categorical exploration ##########################################
independent<-allstate[,2:131]
for (i in 1:114){  #creates barplot of the distribution of the values for each categorical predictor
  barplot(table(independent[,i]), main=names(independent[,i]))
}
for (i in 1:114){   #Prints a list of the levels for each categorical predictor
  print(unique(independent[,i]))
}
for (col in names(independent)){ #2 categorical variables have only one level in the training dataset
  if (nrow(unique(independent[col]))<2){
    print(names(independent[col]))
  }
}  #These two variables cause issue changing to a sparse matrix so I will drop them for  (done up top)
############################################################################################################



########################################continuous exploration #############################################
cormat<-round(cor(allstate[,118:131]), 2)
head(cormat)
melted_cormat<-melt(cormat)
head(melted_cormat)
ggplot(data=melted_cormat, aes(x=X1, y=X2, fill=value)) +
  geom_tile()

high_corr<-melted_cormat[order(abs(melted_cormat$value)),]
############################################################################################################
############################################################################################################




# Transform the loss variable (target) using log transformation
allstate$log_loss <- log(allstate$loss)
allstate <- allstate[ ,-c(132)]

final_prediction$log_loss<-rep(0, nrow(final_prediction))

full<-rbind(allstate, final_prediction)
dim(full)# Create a sparse matrix to handle categorical variables


sparse_full <- sparse.model.matrix(log_loss~.- log_loss, data=full) 
sparse_train<-sparse_full[1:nrow(allstate),]
sparse_test<-sparse_full[(nrow(allstate)+1):nrow(full),]

# Create a separate 
train_log_loss <- allstate$log_loss

set.seed(42)
sample<-sample.int(n=nrow(sparse_train), size=floor(.70*nrow(sparse_train)), replace=FALSE)
train_allstate<-sparse_train[sample,]
train_label<-train_log_loss[sample]
test_allstate<-sparse_train[-sample,]
test_label<-train_log_loss[-sample]
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

#eta tuning
for (i in seq(from=.08, to=.11, by=.005)){
  xgb <- xgboost(data = train_allstate,
                 label = train_label,
                 eta = i,
                 max_depth = 15,
                 gamma = .2,
                 nround=100,
                 subsample = .9,
                 colsample_bytree = .6,
                 objective = "reg:linear",
                 nthread = 3,
                 eval_metric = 'merror',
                 verbose =0)
  ptrain = predict(xgb, sparse_train)
  pvalid = predict(xgb, test_allstate)
  valid<-data.frame(cbind(pvalid,test_label))
  mae<-mean(abs(exp(valid$test_label)-exp(valid$pvalid)))
  print(mae)
}

#nround tuning
for (i in seq(from=50, to=400, by=25)){
  xgb <- xgboost(data = train_allstate,
                 label = train_label,
                 eta = .08,
                 max_depth = 15,
                 gamma = .2,
                 nround=i,
                 subsample = .9,
                 colsample_bytree = .6,
                 objective = "reg:linear",
                 nthread = 3,
                 eval_metric = 'merror',
                 verbose =0)
  ptrain = predict(xgb, sparse_train)
  pvalid = predict(xgb, test_allstate)
  valid<-data.frame(cbind(pvalid,test_label))
  mae<-mean(abs(exp(valid$test_label)-exp(valid$pvalid)))
  print(mae)
}

for (i in seq(from=0, to=20, by=3)){
  xgb <- xgboost(data = train_allstate,
                 label = train_label,
                 eta = .08,
                 max_depth = i,
                 gamma = .2,
                 nround=100,
                 subsample = .9,
                 colsample_bytree = .6,
                 objective = "reg:linear",
                 nthread = 3,
                 eval_metric = 'merror',
                 verbose =0)
  ptrain = predict(xgb, sparse_train)
  pvalid = predict(xgb, test_allstate)
  valid<-data.frame(cbind(pvalid,test_label))
  mae<-mean(abs(exp(valid$test_label)-exp(valid$pvalid)))
  print(mae)
}
for (i in seq(from=0, to=.3, by=.05)){
  xgb <- xgboost(data = train_allstate,
                 label = train_label,
                 eta = .08,
                 max_depth = 10,
                 gamma = i,
                 nround=100,
                 subsample = .9,
                 colsample_bytree = .6,
                 objective = "reg:linear",
                 nthread = 3,
                 eval_metric = 'merror',
                 verbose =0)
  ptrain = predict(xgb, sparse_train)
  pvalid = predict(xgb, test_allstate)
  valid<-data.frame(cbind(pvalid,test_label))
  mae<-mean(abs(exp(valid$test_label)-exp(valid$pvalid)))
  print(mae)
}

xgb <- xgboost(data = train_allstate,
               label = train_label,
               eta = .08,
               max_depth = 10,
               gamma = .2,
               nround=100,
               subsample = .9,
               colsample_bytree = .6,
               objective = "reg:linear",
               nthread = 3,
               eval_metric = 'merror',
               verbose =0)
ptrain = predict(xgb, sparse_train)
pvalid = predict(xgb, test_allstate)
valid<-data.frame(cbind(pvalid,test_label))
mae<-mean(abs(exp(valid$test_label)-exp(valid$pvalid)))
print(mae)


preds <- predict(xgb, sparse_test)
df_preds <- data.frame(id=final_prediction$id, loss = exp(final_predictions))
write.csv(df_preds, 'blue4.csv', row.names = FALSE)
