# Dependances
# install.packages("missMDA")
# install.packages("ggplot2")
#install.packages("ade4")
#
# Challenge
#
library("CatEncoders")

label_encode <- function(X, name) {
  labelEncoder <- LabelEncoder.fit(X[, name])
  z <- transform(labelEncoder,X[, name])
  X[,name]<- z
  return(X)  
}

one_hot_encode <- function(X, name) {
  oneHotEncoder <- OneHotEncoder.fit(data.frame(X[, name]))
  z <- transform(oneHotEncoder,data.frame(X[, name]),sparse=FALSE)
  classes <- slot(slot(oneHotEncoder, "column_encoders")[[1]], "classes")
  colnames(z) <- classes
  X <- cbind(X, z)
  X[,name]<- NULL
  return(X)  
}

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

data <- data_train
ALL_NA <- data[!complete.cases(data),]

# Remove outliers on univariate analysis
# 0.98 car les distributions semblent normales
NO_NA <- na.omit(data)
NO_NA <- NO_NA %>% filter( bmi < quantile(bmi, 0.99))
NO_NA <- NO_NA %>% filter( age < quantile(age, 0.99))
NO_NA <- NO_NA %>% filter( egfr < quantile(egfr, 0.99))
NO_NA <- NO_NA %>% filter( sbp < quantile(sbp, 0.99))
NO_NA <- NO_NA %>% filter( dbp < quantile(dbp, 0.99))
NO_NA <- NO_NA %>% filter( hr < quantile(hr, 0.99))

NO_NA <- NO_NA %>% filter( bmi > quantile(bmi, 0.01))
NO_NA <- NO_NA %>% filter( age > quantile(age, 0.01))
NO_NA <- NO_NA %>% filter( egfr > quantile(egfr, 0.01))
NO_NA <- NO_NA %>% filter( sbp > quantile(sbp, 0.01))
NO_NA <- NO_NA %>% filter( dbp > quantile(dbp, 0.01))
NO_NA <- NO_NA %>% filter( hr > quantile(hr, 0.01))

clean_data <- rbind(NO_NA, ALL_NA)

imputed_data <- missMDA::imputeFAMD(clean_data[, all], ncp = 20)
X_imputed <- imputed_data$completeObs 

X_imputed[, "lvefbin"] <- data_train[, "lvefbin"]

#########
#########
#########


library(SOMbrero)
X_all <- X_imputed
for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X_all <- label_encode(X_all, i)
}


X_all[,quanti] <- scale(X_all[,quanti])

selected <- X_all[,"centre"] == 7 || X_all[,"centre"] == 8 
selected <- !selected


X_all <- one_hot_encode(X_all, "centre")
X_all <- one_hot_encode(X_all, "country")

# c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")
# sup = "egfr", "sbp", "dbp", "hr" "centre", "country", "copd",  "afib"
# X_all[,"lvefbin"] <- NULL
X_all[,"centre"] <- NULL
X_all[,"country"] <- NULL
# X_all[,"copd"] <- NULL
# X_all[,"afib"] <- NULL
# X_all[,"s_dbp"] <- X_all[,"sbp"] - X_all[,"dbp"]
# X_all[,"dbp"] <- NULL
# X_all[,"sbp"] <- NULL

X_all[,"lvefbin"] <- NULL
lvefbin <- c(NO_NA[, "lvefbin"], ALL_NA[,"lvefbin"])
lvefbin[lvefbin == 1] <- 'bad'
lvefbin[lvefbin == 2] <- 'good'


som <- SOMbrero::trainSOM(x.data = X_all[selected,], dimension = c(3, 3), verbose=TRUE, nb.save=50)
plot(som, what="add", type="pie", variable=lvefbin)

# https://cran.r-project.org/web/packages/SOMbrero/vignettes/doc-numericSOM.html
plot(som, what="energy")

plot(som, what="obs", type="hitmap")
plot(som, what="prototypes", type="color", var=1, main="prototypes - x1")
plot(som, what="prototypes", type="lines", print.title=TRUE)
plot(som, what="obs", type="names", print.title=TRUE, scale=c(0.9,0.5))
plot(som, what="prototypes", type="umatrix")
plot(som, type="radar", key.loc=c(-0.5,5), mar=c(0,10,2,0))


write.table(X_all, "clean_data_train_after_som.csv", sep=";", quote=FALSE, row.names = FALSE)

#######

X_all <- X_test
for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X_all <- label_encode(X_all, i)
}

X_all[,quanti] <- scale(X_all[,quanti])

selected <- X_all[,"centre"] == 7

# c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")
# sup = "egfr", "sbp", "dbp", "hr" "centre", "country", "copd",  "afib"
X_all[,"lvefbin"] <- NULL
X_all[,"centre"] <- NULL
X_all[,"country"] <- NULL
X_all[,"copd"] <- NULL
X_all[,"afib"] <- NULL
X_all[,"s_dbp"] <- X_all[,"sbp"] - X_all[,"dbp"]
X_all[,"dbp"] <- NULL
X_all[,"sbp"] <- NULL

write.table(X_all, "clean_data_test_after_som.csv", sep=";", quote=FALSE, row.names = FALSE)


som <- SOMbrero::trainSOM(x.data = X_all, dimension = c(3, 2), verbose=TRUE, nb.save=50)
plot(som, what="add", type="pie", variable=X[,"lvefbin"])

# https://cran.r-project.org/web/packages/SOMbrero/vignettes/doc-numericSOM.html
plot(som, what="energy")

plot(som, what="obs", type="hitmap")
plot(som, what="prototypes", type="color", var=1, main="prototypes - x1")
plot(som, what="prototypes", type="lines", print.title=TRUE)
plot(som, what="obs", type="names", print.title=TRUE, scale=c(0.9,0.5))
plot(som, what="prototypes", type="umatrix")
plot(som, type="radar", key.loc=c(-0.5,5), mar=c(0,10,2,0))


