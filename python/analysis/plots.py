'''
This module contains functions to generate EDA plots
'''

import seaborn as sns
import matplotlib.pyplot as plt

def eda_plots(df, target, numerical_features, categorical_features):
    '''
    Generate plots for EDA

    Parameters
    ----------
    df : pandas DataFrame
        dataset of interest
    target : string
        name of the response
    numerical_features : list of string
        names of the numerical features
    categorical_features : list of string
        names of the categorical features

    Returns
    -------
    None. Show the EDA plots

    '''
    for feature in categorical_features:   
        df.groupby(feature).size().plot.bar();
        plt.show();
    for feature in numerical_features:
        df[feature].plot(kind = 'hist');
        plt.show();
    for feature in categorical_features:
        df.groupby([feature,target]).size().unstack().plot(kind='bar', stacked = True);
        plt.show();
    for feature in numerical_features:   
        sns.violinplot(data = df, x = target, y = feature);
        plt.show();
    
        
