#' Plots for exploratory data analysis
#' @export
#' @import ggplot2
#' @import dplyr
#' @param df data.frame for a given dataset with pre-set of features and a categorical target
#' @param target column name of the target in df
#' @examples {
#' iris$Species <- as.factor(iris$Species)
#' eda_plots(iris, "Species")
#' }
#'
eda_plots <- function(df, target) {

        for(col_name in colnames(df)) {

                if(is.numeric(df[,col_name])) {
                        plt <- ggplot(df, aes_string(x=col_name)) + geom_histogram(fill="steelblue", na.rm = TRUE)
                        print(plt)
                }
                else if(is.factor(df[,col_name])) {
                        plt <- ggplot(df, aes_string(x=col_name, fill=col_name)) + geom_bar()
                        print(plt)
                }
        }

        for(col_name in colnames(df)) {

                if(col_name != target) {
                        if(is.factor(df[,col_name])) {
                                plt <- df %>%
                                        ggplot(aes_string(fill=target, x=col_name)) +
                                        geom_bar(position ="stack") +
                                        guides(fill=guide_legend("survival"))
                                print(plt)
                        }
                        else if(is.numeric(df[,col_name])) {
                                plt <- df %>%
                                        ggplot(aes_string(x=target, y=col_name, fill=target)) +
                                        geom_violin(na.rm=TRUE, adjust=0.5)
                                print(plt)
                        }
                }
        }

}
