
# Import the libraries
library(devtools)
install("./Repositories/analysis")  #your path may be different
library(analysis)

# Load data
titanic <- read.csv("../data/titanic_openml.csv")  #copied the file from practice assignment 9 to here 

# Convert to factors if required, and create the column "family_size"
titanic$survived <- factor(titanic$survived)
titanic$sex <- factor(titanic$sex)
titanic$embarked <- factor(titanic$embarked)
titanic$pclass <- factor(titanic$pclass, order = TRUE, levels = c(1, 2, 3))
titanic$family_size <- titanic$sibsp + titanic$parch + 1
titanic <- titanic[,c('age', 'fare', 'embarked', 'sex', 'pclass', 'family_size', 'survived')]

# Create plots
eda_plots(titanic, "survived")

# Fit the models
fitted_result <- fit_models(titanic, 0.5, "survived", 
                            c('logistic regression', 'tree', 'gradient boosting', 'svm', 
                              'penalised logistic regression','random forest'),
                            log_level = "error")

## Plot to compare the models
plot_model_comparison(fitted_result$task, fitted_result$fitted_models, log_level="error")

