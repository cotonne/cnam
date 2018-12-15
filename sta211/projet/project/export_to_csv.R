quanti <- c("bmi", "age", "egfr", "sbp", "dbp", "hr" )
quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )
all <- c(quali, quanti)

save_to_csv <- function (x, name, add = FALSE) {
  imputed_data <- missMDA::imputeFAMD(x[, all], ncp = 15)
  
  X_imputed <- imputed_data$completeObs 
  
  for(i in c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )) {
    print(i)
    print(table(X_imputed[,i]))
    l <- levels(X_imputed[,i])
    l <- unlist(lapply(l, FUN = function(level) {gsub(".*_([0-9]*)", "\\1", level)}))
    X_imputed[, i] <- as.factor(as.numeric(X_imputed[, i]))
    levels(X_imputed[, i]) <- l
    print(table(X_imputed[,i]))
  }
  
  if(add) {
    X_imputed[, "lvefbin"] <- x[, "lvefbin"]
  }
  write.table(X_imputed, name, sep=";", quote=FALSE)
}

#
# MODIFIEZ MOI
#
setwd("/home/yvan/Documents/cnam/sta211/projet/project/")

file <- "data_train.rda"
load(file)
save_to_csv(data_train, 'data_train_ncp_15.csv', TRUE)

file <- "data_test.rda"
load(file)
save_to_csv(data_test, 'data_test_ncp_15.csv')



