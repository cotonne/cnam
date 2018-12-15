# Dependances
# install.packages("missMDA")
# install.packages("ggplot2")
#install.packages("ade4")
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

#
# MODIFIEZ MOI
#
setwd("/home/yvan/Documents/cnam/sta211/projet/project/")

library("FactoMineR")
library("ade4")
library("PerformanceAnalytics")
library("missMDA")
library(ggplot2)
library(SOMbrero)

## 'data.frame':    987 obs. of  14 variables:
##  01 $ centre      : Factor w/ 22 levels "1","12","13",..: 1 1 1 1 1 1 1 1 1 1 ...
##  02 $ country     : Factor w/ 17 levels "1","10","11",..: 1 1 1 1 1 1 1 1 1 1 ...
##  03 $ gender      : Factor w/ 2 levels "0","1": 2 2 1 2 1 2 1 2 1 2 ...
##  04 $ bmi         : num  26.2 26.7 NA 31.3 NA 26.8 NA 22.7 NA 33.1 ...
##  05 $ age         : int  90 87 84 58 59 71 72 55 72 68 ...
##  06 $ egfr        : num  94.1 38.4 40.9 36.6 164.8 ...
##  07 $ sbp         : int  178 170 104 134 74 134 140 156 157 160 ...
##  08 $ dbp         : int  81 106 59 89 56 51 92 69 102 91 ...
##  09 $ hr          : int  74 102 70 73 95 81 70 102 98 111 ...
##  10 $ copd        : Factor w/ 2 levels "0","1": 1 1 1 2 1 1 1 1 1 1 ...
##  11 $ hypertension: Factor w/ 2 levels "0","1": 2 1 1 1 1 2 1 1 2 1 ...
##  12 $ previoushf  : Factor w/ 2 levels "0","1": 1 1 2 2 1 2 2 1 2 2 ...
##  13 $ afib        : Factor w/ 2 levels "0","1": 1 1 1 2 1 1 1 1 1 1 ...
##  14 $ cad         : Factor w/ 2 levels "0","1": 2 2 2 1 1 2 1 1 2 2 ...
## lvef
## lvefbin : La variable à prédire est la variable binaire lvefbin dont les modalités sont bad et good. 
##          Celle-ci est une version discrète de la variable lvef qui elle est continue. 
##          Une valeur inférieure à 40 de cette variable traduit une insuffisance cardiaque 
##          (correspondant à la modalité bad de la variable lvefbin)
quanti <- c("bmi", "age", "egfr", "sbp", "dbp", "hr" )
quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )
all <- c(quali, quanti)

file <- "data_train.rda"
load(file)

imputed_data <- missMDA::imputeFAMD(data_train[, all], ncp = 15)
X_imputed <- imputed_data$completeObs 

for(i in c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  l <- levels(X_imputed[,i])
  l <- unlist(lapply(l, FUN = function(level) {gsub(".*_([0-9]*)", "\\1", level)}))
  X_imputed[, i] <- as.numeric(X_imputed[, i])
  levels(X_imputed[, i]) <- l
}

X_imputed[, "lvefbin"] <- data_train[, "lvefbin"]

som <- SOMbrero::trainSOM(x.data = X_imputed, dimension = c(10,10))

