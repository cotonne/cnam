#
# Script de préparation des données
#
# Les données sont
#  - Filtrées : on retire le 1er et 99% percentile
#  - Imputées via AFDM. Le nombre de composantes est estimé via estim_ncpFAMD
#  - Les variables centre et country sont remplacées par leurs indicatrices (codage disjonctif complet)
#  - Les autres variables quantitatives sont ecnodées avec 0 ou 1
#  - 


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

# install.packages("dplyr")
# install.packages("onehot")
# install.packages("CatEncoders")
# install.packages("missForest")

quanti <- c("bmi", "age", "egfr", "sbp", "dbp", "hr" )
quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )
all <- c(quali, quanti)

# Filter > Impute > One Hot Encoding > SOM > save

#
# MODIFIEZ MOI
#
setwd("/home/yvan/Documents/cnam/sta211/projet/project/")

library("dplyr")

file <- "data_train.rda"
load(file)

data <- data_train
ALL_NA <- data[!complete.cases(data),]

#################### Filtre / percentile ####################
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

#################### Imputation ####################
# /!\ Ne pas inclure lvefbin => sinon imputation en prenant la variable cible!
a_sample <- sample(nrow(NO_NA), 1000)
# ncp <- missMDA::estim_ncpFAMD(clean_data[a_sample, all], , ncp.max = 50)
# imputed_data <- missMDA::imputeFAMD(clean_data[, all], ncp = 13)
# X_imputed <- imputed_data$completeObs 

imputed_data <- missForest::missForest(clean_data[, all], maxnodes = 200)

#################### Codage ####################
library("CatEncoders")

X <- imputed_data$ximp 

X <- one_hot_encode(X, "centre")
X <- one_hot_encode(X, "country")

for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X <- label_encode(X, i)
}

X[,quanti] <- scale(X[,quanti])

X[,"lvefbin"] <- c(NO_NA[, "lvefbin"], ALL_NA[,"lvefbin"])
X[X[,"lvefbin"] == 1,"lvefbin"] <- 'bad'
X[X[,"lvefbin"] == 2,"lvefbin"] <- 'good'

# X[,quanti] <- scale(X[,quanti])


write.table(X, "clean_data_train.csv", sep=";", quote=FALSE, row.names = FALSE)

#
# TEST
# 
file <- "data_test.rda"
load(file)

#imputed_data <- missMDA::imputeFAMD(data_test, ncp = 20)
imputed_data <- missForest::missForest(data_test, maxnodes = 200)
X_test <- imputed_data$ximp 

X_test <- one_hot_encode(X_test, "centre")
X_test <- one_hot_encode(X_test, "country")

for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X_test <- label_encode(X_test, i)
}

scaled_test <- scale(X_test)
write.table(scaled_test, "clean_data_test.csv", sep=";", quote=FALSE, row.names = FALSE)


