#' Plot and compare the fitted models
#' @export
#' @import ggplot2
#' @import mlr3
#' @import mlr3viz
#' @importFrom lgr get_logger
#' @param task classification task from mlr3
#' @param fitted_models fitted models
#' @param log_level log level to set. Default is NULL, which means no change in log level. See https://mlr3book.mlr-org.com/logging.html for more details
#' @return ggplot
#' @examples {
#' iris$Species <- as.factor(iris$Species)
#' models <- c('logistic regression', 'tree', 'gradient boosting', 'svm',
#' 'penalised logistic regression', 'gradient boosting', 'random forest')
#' fitted_result <- fit_models(iris, 0.5, "Species", models, log_level = 'error')
#' plot_model_comparison(fitted_result$task, fitted_result$fitted_models)
#' }
plot_model_comparison <- function(task, fitted_models, log_level=NULL) {

        # set the log level - optional but default level is 'info', try 'error' or 'warn' for fewer messages
        if(!is.null(log_level)) {
                get_logger("mlr3")$set_threshold(log_level)
                get_logger("bbotk")$set_threshold(log_level)
                get_logger()$set_threshold(log_level)
        }

        set.seed(1) # for reproducible results
        # set the benchmark design and run the comparisons
        bm_design <- benchmark_grid(tasks=task, resamplings=rsmp('cv', folds=3), learners=fitted_models)
        bmr <- benchmark(bm_design, store_models=TRUE)

        measure <- msr('classif.ce')
        bmr$aggregate(measure)

        autoplot(bmr) + scale_x_discrete(labels=names(fitted_models))

}
