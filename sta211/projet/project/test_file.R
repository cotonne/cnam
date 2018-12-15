#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)!=1) {
  stop("Missing test file")
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

file <- "data_test.rda"
load(file)

pred_file = args[1]
read_pred(pred_file)
