- Econometrics
- https://onlinecourses.science.psu.edu/stat857/node/215
- credit scoring : https://cran.r-project.org/doc/contrib/Sharma-CreditScoring.pdf

Cost Matrix ==> Dernière colonne : résultat

This dataset requires use of a cost matrix (see below)


      1        2
----------------------------
  1   0        1
-----------------------
  2   5        0

(1 = Good,  2 = Bad)

the rows represent the actual classification and the columns
the predicted classification.

It is worse to class a customer as good when they are bad (5), 
than it is to class a customer as bad when they are good (1).

==> samples_weight à utiliser?

Notions à faire apparaître
 - Cross Validation
 - Hyper


Analyse du domaine
 - Credit scoring

1. Analyse des données

Base : 
 - https://select-statistics.co.uk/blog/analysing-categorical-survey-data/
 - 
 - Liste des données
   - Unexpected/Incoherent results
 - Table de contigences
 - Analyses univariées : Représentation graphiques
 - Tableau des Variances covariances

 
Nettoyage des données ?
	Categorical to Numeric values
	  Categorical ==> Dummy variables 
	   - https://www.kaggle.com/c/titanic/discussion/5379
	   - http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
	   - http://www.ultravioletanalytics.com/2014/11/05/kaggle-titanic-competition-part-iii-variable-transformations/
	    - Nominal (no order)
	    - Ordinal (with order)
	    Dummy variable trap

Dimension reduction ?
  AFM, AFDM pour les données categorielles
ACP?
Missing values? ==> Aucune

Feature selection : http://sujitpal.blogspot.fr/2013/05/feature-selection-with-scikit-learn.html

Cross-tabulation
Test d'indépendance du X²

2. Regression logistic
 - https://select-statistics.co.uk/blog/analysing-categorical-data-using-logistic-regression-models/

3. SVM

4. Tree
 ==> On garde les données
5. Neural network
  Without hidden layers
  With one layer
  ...?

Conclusion