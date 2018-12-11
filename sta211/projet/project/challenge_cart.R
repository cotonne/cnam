# Dependances
# install.packages("missMDA")
# install.packages("ggplot2")
# install.packages("rpart")
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
library("rpart")

quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )

file <- "data_train.rda"
load(file)

# 15 <= bmi <= 50
# age <= 122
# 
data <- data_train
# data <- data_train[which(15 <= data$bmi), ]
# data <- data_train[which(data$bmi <= 50), ]
# data <- data_train[which(data$age <= 122), ]

#
# Imputation avec FAMD
#
imputed_data <- missMDA::imputeFAMD(data, ncp = 5)
d <- imputed_data$completeObs # [,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad", "lvefbin")]
d[,"lvef"] <- NULL
#
# Modèle: CART
#
fit <- rpart(lvefbin ~  ., method="class", data=d)
plot(fit, uniform=TRUE, main="Classification Tree", branch = 0)
text(fit, use.n=TRUE, all=TRUE, cex=.6)

#
# erreur d'apprentissage
#
pred <- predict(fit, newdata = d, type="class")
erreur_apprentissage <- error_rate(d$lvefbin, pred)
print(erreur_apprentissage)

file_test <- "data_test.rda"
load(file_test)
imputed_data_test <- missMDA::imputeFAMD(data_test, ncp = 5)
d_test <- imputed_data_test$completeObs[,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")]
pred_test <- predict(fit, type = "class", newdata = d_test)
write(pred_test, file = "my_pred.csv")
read_pred("20181123_reg_pred.csv")
