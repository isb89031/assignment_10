
#' Fit the models on the given dataset
#' @export
#' @import mlr3
#' @import mlr3learners
#' @import mlr3pipelines
#' @import mlr3tuning
#' @importFrom lgr get_logger
#' @param df data.frame for a given dataset with pre-set of features and a categorical target
#' @param train_ratio portion of data used for training purpose
#' @param target column name of the target in df
#' @param model_names vector of method names to be used. Choose among 'logistic regression', 'tree', 'gradient boosting', 'svm', 'penalised logistic regression' and 'random forest'
#' @param log_level log level to set. Default is NULL, which means no change in log level. See https://mlr3book.mlr-org.com/logging.html for more details
#' @return list of list: task and a list of fitted models
#' @examples {
#' iris$Species <- as.factor(iris$Species)
#' models <- c('logistic regression', 'tree', 'gradient boosting', 'svm',
#' 'penalised logistic regression', 'gradient boosting', 'random forest')
#' fitted_result <- fit_models(iris, 0.5, "Species", models, log_level='error')
#' }
fit_models <- function(df, train_ratio, target, model_names, log_level=NULL) {

        # set the log level - optional but default level is 'info', try 'error' or 'warn' for fewer messages
        if(!is.null(log_level)) {
                get_logger("mlr3")$set_threshold(log_level)
                get_logger("bbotk")$set_threshold(log_level)
                get_logger()$set_threshold(log_level)
        }

        # set the training and test set
        set.seed(1) # set the seed so that we can reproduce the result
        n <- nrow(df)
        train_set <- sample(n, round(train_ratio*n))
        test_set <- setdiff(1:n, train_set)

        # set the task
        task <- TaskClassif$new('analysis', backend=df, target=target)

        # fit the models
        model_mapping <- c('logistic regression' = "classif.log_reg",
                           'tree' = "classif.rpart",
                           'gradient boosting' = "classif.xgboost",
                           'svm' = "classif.svm")

        fitted_models <- list()
        for (model_name in model_names) {
                print(model_name)
                if (model_name %in% names(model_mapping)) {
                        fitted_models[[model_name]] <- my_model(task, train_set, model_mapping[[model_name]])
                } else if (model_name == 'penalised logistic regression') {
                        fitted_models[[model_name]] <- my_plr(task, train_set)
                } else if (model_name == 'random forest') {
                        fitted_models[[model_name]] <- my_rf(task, train_set)
                } else {
                        stop(paste(model_name, "is not supported"))
                }
        }
        return (list(task=task, fitted_models=fitted_models))
}


#' Fit model using any of the following algorithms: logistic regression, tree, gradient boosting, svm
#' @description This function is not exported
#' @import mlr3
my_model <- function(task, train_set, classifier_name) {

        if(classifier_name %in% c("classif.log_reg", "classif.rpart")) {
                convert.factor <- FALSE
        } else if(classifier_name %in% c("classif.xgboost", "classif.svm") ) {
                convert.factor <- TRUE
        } else {
                stop(paste(classifier_name, "is not supported"))
        }

        learner <- lrn(classifier_name)

        if(convert.factor) {
                fencoder <- po("encode", method = "treatment",
                               affect_columns = selector_type("factor"))
                ord_to_int <- po("colapply", applicator = as.integer,
                                 affect_columns = selector_type("ordered"))
                gc <- po('imputemean') %>>%
                        fencoder %>>% ord_to_int %>>%
                        po(learner)
        } else {
                gc <- po('imputemean') %>>%
                        po(learner)
        }

        glrn <- GraphLearner$new(gc)
        glrn$train(task, row_ids = train_set)

        return (glrn)
}


#' Penalised logistic regression - includes tuning hyperparameters
#' @description This function is not exported
#' @import mlr3
#' @import paradox
my_plr <- function(task, train_set) {

        learner_plr <- lrn('classif.glmnet')

        fencoder <- po("encode", method = "treatment",
                       affect_columns = selector_type("factor"))
        ord_to_int <- po("colapply", applicator = as.integer,
                         affect_columns = selector_type("ordered"))

        gc_plr <- po('scale') %>>%
                fencoder %>>% ord_to_int %>>%
                po('imputemean') %>>%
                po(learner_plr)

        glrn_plr <- GraphLearner$new(gc_plr)

        # for tuning
        tuner <- tnr('grid_search')
        terminator <- trm('evals', n_evals = 20)
        tune_lambda <- ParamSet$new (list(
                ParamDbl$new('classif.glmnet.lambda', lower = 0.001, upper = 2)
        ))

        at_plr <- AutoTuner$new(
                learner = glrn_plr,
                resampling = rsmp('cv', folds = 3),
                measure = msr('classif.ce'),
                search_space = tune_lambda,
                terminator = terminator,
                tuner = tuner
        )
        at_plr$train(task, row_ids = train_set)

        return (at_plr)
}


#' Random forest - includes tuning hyperparameters
#' @import mlr3
#' @import paradox
my_rf <- function(task, train_set) {

        learner_rf <- lrn('classif.ranger')
        learner_rf$param_set$values <- list(min.node.size = 4)

        gc_rf <- po('scale') %>>%
                po('imputemean') %>>%
                po(learner_rf)
        glrn_rf <- GraphLearner$new(gc_rf)

        # for tuning
        tuner <- tnr('grid_search')
        terminator <- trm('evals', n_evals = 20)
        tune_ntrees <- ParamSet$new (list(
                ParamInt$new('classif.ranger.num.trees', lower = 50, upper = 600)
        ))
        at_rf <- AutoTuner$new(
                learner = glrn_rf,
                resampling = rsmp('cv', folds = 3),
                measure = msr('classif.ce'),
                search_space = tune_ntrees,
                terminator = terminator,
                tuner = tuner
        )
        at_rf$train(task, row_ids = train_set)

        return (at_rf)
}

