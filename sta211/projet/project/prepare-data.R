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

# /!\ Ne pas inclure lvefbin => sinon imputation en prenant la variable cible!
imputed_data <- missMDA::imputeFAMD(clean_data[, all], ncp = 20)
X_imputed <- imputed_data$completeObs 

library("CatEncoders")

X <- X_imputed

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

imputed_data <- missMDA::imputeFAMD(data_test, ncp = 20)
X_test <- imputed_data$completeObs 

X_test <- one_hot_encode(X_test, "centre")
X_test <- one_hot_encode(X_test, "country")

for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X_test <- label_encode(X_test, i)
}

scaled_test <- scale(X_test)
write.table(scaled_test, "clean_data_test.csv", sep=";", quote=FALSE, row.names = FALSE)


library(SOMbrero)
X_all <- X_imputed
for(i in c("gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
  X_all <- label_encode(X_all, i)
}

X_all[,quanti] <- scale(X_all[,quanti])


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



som <- SOMbrero::trainSOM(x.data = X_all, dimension = c(3, 3), verbose=TRUE, nb.save=50)
plot(som, what="add", type="pie", variable=X[,"lvefbin"])

# https://cran.r-project.org/web/packages/SOMbrero/vignettes/doc-numericSOM.html
plot(som, what="energy")

plot(som, what="obs", type="hitmap")
plot(som, what="prototypes", type="color", var=1, main="prototypes - x1")
plot(som, what="prototypes", type="lines", print.title=TRUE)
plot(som, what="obs", type="names", print.title=TRUE, scale=c(0.9,0.5))
plot(som, what="prototypes", type="umatrix")
plot(som, type="radar", key.loc=c(-0.5,5), mar=c(0,10,2,0))


