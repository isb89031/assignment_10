% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/eda_plots.R
\name{eda_plots}
\alias{eda_plots}
\title{#' Plots for exploratory data analysis}
\usage{
eda_plots(df, target)
}
\arguments{
\item{df}{data.frame for a given dataset with pre-set of features and a categorical target}

\item{target}{column name of the target in df}
}
\description{
#' Plots for exploratory data analysis
}
\examples{
{
iris$Species <- as.factor(iris$Species)
eda_plots(iris, "Species")
}

}
