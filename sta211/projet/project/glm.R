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

taux_erreur <- function(data, pred) {
  table_confusion <- table(data, pred)
  1.0 - (table_confusion[1,1] + table_confusion[2,2])/sum(table_confusion)
}

setwd("/home/yvan/Documents/cnam/sta211/projet/project/")

library("FactoMineR")
library("ade4")
library("PerformanceAnalytics")
library("missMDA")
library(ggplot2)

file <- "data_train.rda"
load(file)

file_test <- "data_test.rda"
load(file_test)
X_test <- data_test

library("dplyr")
X <- data_train

#
# Suppression des variables redondantes
#
# X[,"egfr"] <- NULL
# X[,"hypertension"] <- NULL
# X_test[,"egfr"] <- NULL
# X_test[,"hypertension"] <- NULL

#
# Suppression premier et dernier centile
#
ALL_NA <- X[!complete.cases(X),]
NO_NA <- na.omit(X)
NO_NA <- NO_NA %>% filter( bmi < quantile(bmi, 0.99))
NO_NA <- NO_NA %>% filter( age < quantile(age, 0.99))
NO_NA <- NO_NA %>% filter( sbp < quantile(sbp, 0.99))
NO_NA <- NO_NA %>% filter( dbp < quantile(dbp, 0.99))
NO_NA <- NO_NA %>% filter( hr < quantile(hr, 0.99))

NO_NA <- NO_NA %>% filter( bmi > quantile(bmi, 0.01))
NO_NA <- NO_NA %>% filter( age > quantile(age, 0.01))
NO_NA <- NO_NA %>% filter( sbp > quantile(sbp, 0.01))
NO_NA <- NO_NA %>% filter( dbp > quantile(dbp, 0.01))
NO_NA <- NO_NA %>% filter( hr > quantile(hr, 0.01))

X <- rbind(NO_NA, ALL_NA)

y <- "lvefbin"
lvefbin <- X[y]
X["lvefbin"] <- NULL
X["lvef"] <- NULL

#
# Création de la variable centre_country
#
X[, "centre_country"] <- with(X, interaction(centre, country), drop = TRUE )
X[,"centre"] <- NULL
X[,"country"] <- NULL

X_test[, "centre_country"] <- with(X_test, interaction(centre, country), drop = TRUE )
X_test[,"centre"] <- NULL
X_test[,"country"] <- NULL

#
# Normalisation variable quantitative
#
X[,c("bmi", "age", "sbp", "dbp", "hr")] <- scale(X[,c("bmi", "age", "sbp", "dbp", "hr")])
X_test[,c("bmi", "age", "sbp", "dbp", "hr")] <- scale(X_test[,c("bmi", "age", "sbp", "dbp", "hr")])

#
# Imputation des données
#
# all <- rbind(X, X_test)
imputed_data <- missMDA::imputeFAMD(X, ncp = 25, method = "Regularized")
d <- imputed_data$completeObs

#
# SOM
#
#library(SOMbrero)
#som <- SOMbrero::trainSOM(x.data = tmp_d, dimension = c(3, 3), verbose=TRUE, nb.save=50)
#d[,"cluster_som"] <- factor(som$clustering)

#
# Regression logistique
#
d[,y] <- lvefbin
write.table(d, "clean_data_train.csv", sep=";", quote=FALSE, row.names = FALSE)

reg <- glm(lvefbin ~ ., data=d, family = binomial(logit))
# reg <- lm(lvef ~ ., data=d)
pred <- predict(reg, type = "response", newdata = d)
erreur_apprentissage <- taux_erreur(d$lvefbin, pred > 0.5)

#
# PREDICTION
#


#
# Imputation des données
#
imputed_data_test <- missMDA::imputeFAMD(X_test, ncp = 25, method = "Regularized")

write.table(imputed_data_test$completeObs, "clean_data_test.csv", sep=";", quote=FALSE, row.names = FALSE)
#
# Prediction
#
pred_test <- predict(reg, type = "response", newdata = imputed_data_test$completeObs)
res_test <- pred_test > 0.5
res_test[res_test == TRUE] <- "good"
res_test[res_test == FALSE] <- "bad"

#
# Verification fichier
#
predicted_file_name <- paste(format(Sys.time(), format="%Y%m%d%H%M"), "glm.csv", sep = "-")
write(res_test, file = predicted_file_name)
read_pred(predicted_file_name)
