# Dependances
# install.packages("missMDA")
# install.packages("ggplot2")
# install.packages("rpart")
# install.packages("dismo")
# install.packages("caret")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("inTrees")
#
# Challenge
#
read_pred <- function(file, n = nrow(data_test)) { 
  y_pred <- scan(file, what = "character") 
  y_pred <- factor(y_pred, levels = c("bad", "good")) 
  if (length(y_pred) != n) 
    stop("incorrect number of predictions") 
  if (any(is.na(y_pred))) 
    stop("predictions contain missing values (NA)") 
  return(y_pred) 
}

findIndex <- function(df, colnames) {
  unlist(lapply(X = colnames, FUN = function(x) grep(paste(c("^", x, "$"), collapse = ""), names(df))))
}

taux_erreur <- function(data, pred) {
  table_confusion <- table(data, pred)
  1.0 - (table_confusion[1,1] + table_confusion[2,2])/sum(table_confusion)
}
error_rate <- 
  function(y_pred, y_test){ 
    FP = (y_pred == "bad") & (y_test == "good") 
    FN = (y_pred == "good") & (y_test == "bad") 
    return(sum(FP+FN)/length(y_test)) 
  }

#
# MODIFIEZ MOI
#
setwd("/home/yvan/Documents/cnam/sta211/projet/project/")

library("FactoMineR")
library("ade4")
library("PerformanceAnalytics")
library("missMDA")
library(ggplot2)
library("randomForest")
library("dismo") #kfold
library("caret") # computing cross-validation methods
library("e1071") # require for CV with random forests
library("inTrees") # rfRules

quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )

file <- "data_train.rda"
load(file)

# 15 <= bmi <= 50
# age <= 122
# 
data <- data_train
# data <- data_train[which(15 <= data$bmi), ]
# data <- data_train[which(data$bmi <= 50), ]
#data <- data_train[which(data$age <= 122), ]

#
# Imputation avec FAMD
#
imputed_data <- missMDA::imputeFAMD(data, ncp = 5)
d <- imputed_data$completeObs # [,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad", "lvefbin")]
d[,"lvef"] <- NULL
# 
# Cross-validation
# Modèle: RANDOM-FOREST
#
train.control <- trainControl(method = "cv", number = 10)
# model <- caret::train(lvefbin ~., data = d, method = "rf", trControl = train.control)
model <- caret::train(lvefbin ~., data = d, method = "rfRules", trControl = train.control,
               ntree=50, maxdepth=5)
# model <- randomForest(lvefbin ~ ., method="class", data=d)

#
#
variables <- c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")
X  <- d[,variables]
y  <- d$lvefbin

#
# erreur d'apprentissage
#
pred <- predict(model, newdata = d, type="raw")
erreur_apprentissage <- error_rate(d$lvefbin, pred)
print(erreur_apprentissage)
varImp(model)

file_test <- "data_test.rda"
load(file_test)
imputed_data_test <- missMDA::imputeFAMD(data_test, ncp = 5)
d_test <- imputed_data_test$completeObs # [,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")]
pred_test <- predict(model, type = "raw", newdata = d_test)
write(as.character(pred_test), file = "20181126_kfold_rf_pred.csv")
read_pred("20181126_kfold_rf_pred.csv")
