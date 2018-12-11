# A1 (quali) :  Status of existing checking account
# A11 :      ... <    0 DM
# A12 : 0 <= ... <  200 DM
# A13 :      ... >= 200 DM / salary assignments for at least 1 year
# A14 : no checking account
# 
# A2 (numerical) : Duration in month
# 
# A3 (quali) : Credit history
# A30 : no credits taken/all credits paid back duly
# A31 : all credits at this bank paid back duly
# A32 : existing credits paid back duly till now
# A33 : delay in paying off in the past
# A34 : critical account/other credits existing (not at this bank)
# 
# A4 (quali) : Purpose
# A40 : car (new)
# A41 : car (used)
# A42 : furniture/equipment
# A43 : radio/television
# A44 : domestic appliances
# A45 : repairs
# A46 : education
# A47 : (vacation - does not exist?)
# A48 : retraining
# A49 : business
# A410 : others
# 
# A5 (numerical) : Credit amount
# 
# A6 (qualitative) : Savings account/bonds
# A61 :          ... <  100 DM
# A62 :   100 <= ... <  500 DM
# A63 :   500 <= ... < 1000 DM
# A64 :          .. >= 1000 DM
# A65 :   unknown/ no savings account
# 
# A7 (qualitative) : Present employment since
# A71 : unemployed
# A72 :       ... < 1 year
# A73 : 1  <= ... < 4 years  
# A74 : 4  <= ... < 7 years
# A75 :       .. >= 7 years
# 
# A8:  (numerical) : Installment rate in percentage of disposable income
# 
# A9:  (qualitative) : Personal status and sex
# A91 : male   : divorced/separated
# A92 : female : divorced/separated/married
# A93 : male   : single
# A94 : male   : married/widowed
# A95 : female : single
# 
# A10: (qualitative) : Other debtors / guarantors
# A101 : none
# A102 : co-applicant
# A103 : guarantor
# 
# A11: (numerical) : Present residence since
# 
# A12: (qualitative) : Property
# A121 : real estate
# A122 : if not A121 : building society savings agreement/ life insurance
# A123 : if not A121/A122 : car or other, not in A6
# A124 : unknown / no property
# 
# A13: (numerical) : Age in years
# 
# A14: (qualitative) : Other installment plans 
# A141 : bank
# A142 : stores
# A143 : none
# 
# A15: (qualitative) : Housing
# A151 : rent
# A152 : own
# A153 : for free
# 
# A16: (numerical) : Number of existing credits at this bank
# 
# A17: (qualitative) : Job
# A171 : unemployed/ unskilled  - non-resident
# A172 : unskilled - resident
# A173 : skilled employee / official
# A174 : management/ self-employed/highly qualified employee/ officer
# 
# A18: (numerical) : Number of people being liable to provide maintenance for
# 
# A19: (qualitative) : Telephone
# A191 : none
# A192 : yes, registered under the customers name
# 
# A20: (qualitative) : foreign worker
# A201 : yes
# A202 : no

# install.packages("FactoMineR")
# install.packages("ade4")
# install.packages("corrplot")

#
# Fonction Rapport de corrélation
# 
vartot <- function(x) {
  res <- sum((x - mean(x))^2)
  return(res)
}

varinter <- function(x, gpe) {
  moyennes <- tapply(x, gpe, mean)
  effectifs <- tapply(x, gpe, length)
  res <- (sum(effectifs * (moyennes - mean(x))^2))
  return(res)
}

eta2 <- function(x, gpe) {
  res <- varinter(x, gpe)/vartot(x)
  return(res)
}

###############################

setwd("/home/yvan/Documents/cnam/sta211/projet/project/")
library("FactoMineR")
library("ade4")
library("PerformanceAnalytics")

quanti <- c(2, 5, 8, 11, 13, 16, 18 )
quali <- c(1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20)

file <- "german.data.csv"
all_data <- read.table(file, sep = " ",header = F, na.strings ="", stringsAsFactors= F, 
                   colClasses=c('factor', 'numeric', 'factor', 
                                'factor', 'numeric', 'factor', 
                                'factor', 'numeric', 'factor',
                                'factor', 'numeric', 'factor',
                                'numeric', 'factor', 'factor',
                                'numeric', 'factor', 'numeric',
                                'factor', 'factor'))

X <- all_data[,1:20]
Y <- all_data[,21]

for(i in quali){
  tbl <- table(X[,i], Y)
  res <- chisq.test(tbl) 
  if(res$p.value >= 0.05) {
    print(paste(c(i, "& Y non indepedendants avec un risque d'erreur p <= 5%, (p-value=", res$p.value, ")"), collapse=" "))
  }
}

for(i in quanti) {
  echantillon <- sample(nrow(X), 200)
  Xe <- X[echantillon,i]
  Ye <- Y[echantillon]
  print(paste(c("Rapport de correlation entre", i, "et Y =", eta2(Xe, Ye))), collapse=" ")
}

# Echantillonage
data <- X[sample(nrow(X), 200), ]
data <- X

# Lien entre variables quanti
chart.Correlation(subset(data, select=quanti))

# Lien entre variables quali
prop.table(table(X$V1, X$V3))
margin.table(table(X$V1, X$V3), 1)

for(i in seq(1,length(quali) -1)) {
  for(j in seq(i+1, length(quali))) {
    u <- quali[i]
    v <- quali[j]
    # Tableau de contingence
    tbl <- table(data[,u], data[,v])
    # Test du χ² d'indépendance
    # (H0) = les deux variables X et Y sont indépendantes.
    # (H1) = les deux variables ne sont pas indépendantes
    # L'hypothèse (H0) peut être rejeté avec un risque d'erreur de p < 5%
    res <- chisq.test(tbl) 
    if(res$p.value >= 0.05) {
      print(paste(c(i, "&", j, " non indepedendants avec un risque d'erreur p <= 5%, (p-value=", res$p.value, ")"), collapse=" "))
    }
  }
}

# Lien entre la variable à expliquer et le montant
group <- as.numeric(X$V2)
fit <- aov(group ~ Y, data.frame(group, Y))
summary(fit)
bartlett.test(group ~ Y)
plot(group ~ Y, data = data.frame(group, Y))

# Lien entre variables quali & quanti
# (H0) = {le facteur n’a pas d’effet sur Y } 
# (H1) = {il existe un effet du facteur sur Y}
group <- data$V1
fit <- aov(data$V2 ~ data$V1, data[,1:2])
summary(fit)

# Tester l’hypothèse de homoscédasticité : Test de Bartlett
# (H0) = { σ²_1 = σ²_2 = ... = σ²_I }
# (H1) = { ∃i,j : σ²_i ≠ σ²_j }
bartlett.test(data$V2 ~ data$V1)

group <- all_data$V1
fit <- aov(all_data$V2 ~ all_data$V1, all_data[,1:2])
summary(fit)

plot(all_data$V2 ~ all_data$V1, data = all_data[,1:2])

# Boxplot
outliers = boxplot(data[,2], plot=FALSE)$out

# Histogramme
barplot(table(data[,3]))

# Discretization
cut(mydata$Age, seq(0,30,5), right=FALSE, labels=c(1:6))

#
# Analyse factorielle de données mixtes
#
res <- FAMD(data)

barplot(res$eig[, 2], main="Eigen vectors - selected vectors are in black", names.arg = paste("dim", seq(1, nrow(res$eig))))

# res.mca = MCA(tea, quanti.sup=19, quali.sup=c(20:36))

# Codage Disjonctif Complet
bd <- acm.disjonctif(data)
bds <- scale(bd)

#
# Cartes de Kohonen
#
install.packages("SOMbrero")
library("SOMbrero")

my.som <- trainSOM(x.data=bds, dimension=c(5,5), nb.save=10, maxit=2000, 
                   scaling="none", radius.type="letremy")

plot(my.som, what="energy")

# Clustering
plot(my.som, what="obs", type="hitmap")

# Clustering interpretation
plot(my.som, what="prototypes", type="lines", print.title=TRUE)
plot(my.som, what="obs", type="barplot", print.title=TRUE)
plot(my.som, what="obs", type="radar", key.loc=c(-0.5,5), mar=c(0,10,2,0))
plot(my.som, what="prototypes", type="umatrix")

