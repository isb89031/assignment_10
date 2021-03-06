% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_models.R
\name{fit_models}
\alias{fit_models}
\title{Fit the models on the given dataset}
\usage{
fit_models(df, train_ratio, target, model_names, log_level = NULL)
}
\arguments{
\item{df}{data.frame for a given dataset with pre-set of features and a categorical target}

\item{train_ratio}{portion of data used for training purpose}

\item{target}{column name of the target in df}

\item{model_names}{vector of method names to be used. Choose among 'logistic regression', 'tree', 'gradient boosting', 'svm', 'penalised logistic regression' and 'random forest'}

\item{log_level}{log level to set. Default is NULL, which means no change in log level. See https://mlr3book.mlr-org.com/logging.html for more details}
}
\value{
list of list: task and a list of fitted models
}
\description{
Fit the models on the given dataset
}
\examples{
{
iris$Species <- as.factor(iris$Species)
models <- c('logistic regression', 'tree', 'gradient boosting', 'svm',
'penalised logistic regression', 'gradient boosting', 'random forest')
fitted_result <- fit_models(iris, 0.5, "Species", models, log_level='error')
}
}
