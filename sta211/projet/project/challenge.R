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

file <- "data_train.rda"
load(file)

#
# Données manquantes
# 
for(i in append(quanti, quali)) {
  number_of_missigin_values <- sum(is.na(data_train[, i]))
  vector_size <- length(data_train[,i])
  percent <- round(100 * number_of_missigin_values/vector_size, 2)
  print(paste(c(i, percent, "%"), collapse = " "))
}

summary(data_train[is.na(data_train$bmi),])

#
# Echantillonage
#
data_train <- data_train[which(15 <= data$bmi), ]
data_train <- data_train[which(data$bmi <= 50), ]
data_train <- data_train[which(data$age <= 122), ]
NO_NA <- na.omit(data_train)
a_sample <- sample(nrow(NO_NA), 400)
X <- data_train
X <- NO_NA[a_sample,]
Y <- NO_NA[a_sample,"lvef"]

plot(X[, c("gender", "hypertension")])

hist(Y)

# Lien entre variables quanti
chart.Correlation(X[,quanti])

chart.Correlation(X[,quanti], method = "spearman")

# test de spearman
for(i in seq(1,length(quanti) -1)) {
  for(j in seq(i+1, length(quanti))) {
    u <- quanti[i]
    v <- quanti[j]
    res <- cor.test(X[,u], X[,v], method = "spearman")
    if(res$p.value >= 0.05) {
      print(paste(c(u, "&", v, " indepedendants avec un risque d'erreur p <= 5%, (p-value=", 
                    res$p.value, "rho = ", res$estimate,")"), collapse=" "))
    }
  }
}

# Lien entre variable quali
for(i in seq(1,length(quali) -1)) {
  for(j in seq(i+1, length(quali))) {
    u <- quali[i]
    v <- quali[j]
    # Tableau de contingence
    tbl <- table(X[,u], X[,v])
    # Test du χ² d'indépendance
    # (H0) = les deux variables X et Y sont indépendantes.
    # (H1) = les deux variables ne sont pas indépendantes
    # L'hypothèse (H0) peut être rejeté avec un risque d'erreur de p < 5%
    res <- chisq.test(tbl) 
    res
    if(res$p.value <= 0.05) {
      print(paste(c(u, "&", v, " indepedendants avec un risque d'erreur p <= 5%, (p-value=", res$p.value, ")"), collapse=" "))
    }
  }
}

# Lien entre variables quali & quanti
# (H0) = {le facteur n’a pas d’effet sur Y } 
# (H1) = {il existe un effet du facteur sur Y}
# (H0) est rejetée si la probabilité est plus petite que le seuil α
for(i in quali) {
  for(j in quanti) {
    group <- X[, i]
    fit <- aov(X[,j] ~ X[,i], X[,c(i, j)])
    print(paste(i, "-", j))
    print(summary(fit))
  }
}

#
# Lien des variables
# Avec variable à expliquer
#
chart.Correlation(data.frame(X[,quanti], Y))

for(i in quali) {
  group <- X[, i]
  fit <- aov(Y ~ X[,i], X[,c(i, "lvef")])
  print(paste(i, "- lvef"))
  print(summary(fit))
}

plot(X[, c("country", "sbp")])
# list rows of data that have missing values
# mydata[!complete.cases(mydata),]

# create new dataset without missing data 
# newdata <- na.omit(mydata)

for(i in quali) {
  # barplot(table(data_train[,i]), main = i)  
  barplot(as.matrix(table(data_train[,i]) / length(data_train[,i])))
}

m <- as.matrix(na.omit(data_train[,quali]))
v <- apply(m, 2, as.numeric)
p <- margin.table(v, 2)
boxplot(p)

# dbp supplémentaires car redondantes avec sbp
# c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")
index_des_variables_supplementaires <- findIndex(X, c("centre", "country", "lvef", "lvefbin"))
res <- FAMD(X, ncp = 5, sup.var = index_des_variables_supplementaires)
res <- FAMD(X, ncp = 5, sup.var = index_des_variables_supplementaires)

plot(res)
plot.FAMD(res, choix = c("quanti"), axes = c(1, 2))
plot.FAMD(res, choix = c("quanti"), axes = c(3, 4))
plot.FAMD(res, choix = c("quanti"), axes = c(1, 3))
plot.FAMD(res, choix = c("quali"), axes = c(1, 3))
plot.FAMD(res, choix = c("ind"), lab.ind = FALSE, axes = c(1, 3))
plot.FAMD(res, choix = c("quanti"), axes = c(2, 3))

imputed_data <- missMDA::imputeFAMD(data_train[,c(quali, quanti)], ncp = 15)

# Retrait des variables non correlées
index_des_variables_supplementaires <- findIndex(X, c("centre", "country", "lvef", "lvefbin", "egfr",  "afib",  "copd"))
FAMD(imputed_data$completeObs, sup.var = index_des_variables_supplementaires)

#
# Analyse multi-tableaux
#
index_des_variables_supplementaires_quali <- findIndex(X, c("centre", "country", "lvefbin",  "afib",  "copd"))
index_des_variables_supplementaires_quanti <- findIndex(X, c("lvef", "egfr"))
w <- imputed_data$completeObs[,c("bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad", "gender")]
for(i in c("bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")) {
  class(w[,i]) <- "numeric"
}
res <- DMFA(w)

#
# AFD
#
imputed_data <- missMDA::imputeFAMD(data_train, ncp = 5)
d <- imputed_data$completeObs[,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad", "lvefbin")]
library(MASS)
fit <- lda(lvefbin ~ ., data=d)
values <- predict(fit)
plot(values$x[,1],values$x[,2])

#
# Regression logistique
#
reg <- glm(lvefbin ~ ., data=d, family = binomial(logit))
pred <- predict(reg, type = "response", newdata = d)
erreur_apprentissage <- taux_erreur(d$lvefbin, pred > 0.5)

file_test <- "data_test.rda"
load(file_test)
imputed_data_test <- missMDA::imputeFAMD(data_test, ncp = 5)
d_test <- imputed_data_test$completeObs[,c("gender", "bmi", "age", "sbp", "hr", "hypertension", "previoushf", "cad")]
pred_test <- predict(reg, type = "response", newdata = d_test)
res_test <- pred_test > 0.5
res_test[res_test == TRUE] <- "good"
res_test[res_test == FALSE] <- "bad"
write(res_test, file = "my_pred.csv")
read_pred("20181123_reg_pred.csv")
read_pred("python_mlp.csv")
