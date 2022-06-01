'''
This module contains functions to fit models on datasets with categorical responses and compare the performance of different models
'''

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV      
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types


'''
'''

def fit_model(X_train, y_train, models, numerical_features, categorical_features):
    '''
    This function fits the training data to the given models

    Parameters
    ----------
    X_train : pandas DataFrame
        Training data (features)
    y_train : pandas Series
        Training data (categorical response)
    models : container of string
        names of the model to fit. Only support 'logistic regression', 'tree', 'gradient boosting', 'svm', 'penalised logistic regression', 'gradient boosting', 'random forest' and 'gaussian process'
    numerical_features : list of string
        names of numerical features in X_train
    categorical_features : list of string
        names of categorical_features in X_train
    Returns
    -------
    fitted_models : dict of fitted models

    '''
    
    
    # Applying SimpleImputer and StandardScaler into a pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])
    
    # Applying SimpleImputer and then OneHotEncoder into another pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    data_transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)]) 
    
    available_models = {'logistic regression': Pipeline(steps=[('data_transformer', data_transformer),
                                                               ('pipe_lr', LogisticRegression(max_iter=10000,penalty='none'))]),
                        'gradient boosting': Pipeline(steps=[('data_transformer', data_transformer), 
                                                             ('pipe_gdb',GradientBoostingClassifier(random_state=2))]),
                        'tree': Pipeline(steps=[('data_transformer', data_transformer),
                                                ('pipe_tree', DecisionTreeClassifier(random_state=0))]),
                        
                        'svm': Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_svm',  LinearSVC(random_state=0, max_iter=10000, tol=0.01))]),
                        'penalised logistic regression': Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_plr', LogisticRegression(penalty='l1', max_iter=10000, tol=0.01, solver='saga'))]),
                        'gaussian process': Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_gp',  GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0))]),
                        'random forest': Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_rf', RandomForestClassifier(random_state=0))])}
    
    param_grid = {
        'data_transformer__numerical__imputer__strategy': ['mean', 'median'],
        'data_transformer__categorical__imputer__strategy': ['constant','most_frequent']
    }
    fitted_models = {}
    for model in models:
        if model not in available_models:
            KeyError(f"{model} is not available.")
        pipe = available_models[model]
        grid_lr = GridSearchCV(pipe, param_grid=param_grid)
        fitted_models[model] = grid_lr.fit(X_train, y_train)
    return fitted_models

def comparison_plot(fitted_models, X_test, y_test):
    '''
    This function compares the performance of different models using the ROC plot (with AUC)

    Parameters
    ----------
    fitted_models : dict of fitted models from fit_model()
    X_test : pandas DataFrame
        Test data (features)
    y_test : pandas Series
        Test data (categorical response)

    Returns
    -------
    None. Plot ROC curves

    '''
    ax = plt.gca()
    for mod_name, fitted_model in fitted_models.items():
        plot_roc_curve(fitted_model, X_test, y_test, ax=ax, name=mod_name)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.show()
