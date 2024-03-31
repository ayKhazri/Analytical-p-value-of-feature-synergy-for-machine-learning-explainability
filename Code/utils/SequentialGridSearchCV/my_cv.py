import numpy as np
from sklearn.metrics import mean_squared_error

#------------------# PARAMETERS #-------------------------------------
EARLY_STOPPING_ROUNDS = 100

#---------------------------------------------------------------------

def my_cv(df, predictors, response, kfolds, regressor, verbose=False):
    """
    Roll our own CV 
    train each kfold with early stopping
    return average metric, sd over kfolds, average best round
    """

    global EARLY_STOPPING_ROUNDS

    metrics = []
    best_iterations = []

    for train_fold, cv_fold in kfolds.split(df):

        fold_X_train = df[predictors].values[train_fold]
        fold_y_train = df[response].values[train_fold]
        fold_X_test = df[predictors].values[cv_fold]
        fold_y_test = df[response].values[cv_fold]

        regressor.fit(fold_X_train, fold_y_train,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      eval_set=[(fold_X_test, fold_y_test)],
                      eval_metric='rmse',
                      verbose=verbose
                    )
        
        y_pred_test = regressor.predict(fold_X_test)
        metrics.append(np.sqrt(mean_squared_error(fold_y_test, y_pred_test)))
        best_iterations.append(regressor.best_iteration)
        
    return np.average(metrics), np.std(metrics), np.average(best_iterations)