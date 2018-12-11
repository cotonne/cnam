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

quanti <- c("bmi", "age", "egfr", "sbp", "dbp", "hr" )
quali <- c("centre", "country", "gender", "copd", "hypertension", "previoushf", "afib", "cad" )

file <- "data_train.rda"
load(file)

names <- colnames(data_train)

for(i in seq(1,length(names) -1)) {
  ni <- names[i]
  index_ni <- which(is.na(data_train[,ni]))
  length_i <-length(index_ni)
  if(length_i ==0) 
    next
  
  for(j in seq(i+1, length(names))) {
    nj <- names[j]
    # P(nj == NA/ ni == NA)
    index_nj <- which(is.na(data_train[,nj]))
    p <- length(intersect(index_ni, index_nj)) / length_i
    if(p > .4) {
      print(sprintf("P(%s/%s) = %.0f%% (nb = %i)", nj, ni, 100 * p, length_i))
    }
  }
}
