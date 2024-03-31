import datetime
import pandas as pd
from utils.SequentialGridSearchCV.my_cv import my_cv
from xgboost import XGBRegressor

#------------------# PARAMETERS #-------------------------------------
N_ESTIMATORS = 1000000

#---------------------------------------------------------------------

def cv_over_param_dict(df, param_dict, predictors, response, kfolds, verbose=False):
    """
    given a list of dictionaries of xgb params
    run my_cv on params, store result in array
    return updated param_dict, results dataframe
    """

    global N_ESTIMATORS

    start_time = datetime.datetime.now()
    print("%-20s %s" % ("Start Time", start_time))

    results = []

    for i, d in enumerate(param_dict):
        xgb = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=N_ESTIMATORS, 
            verbosity=1,
            n_jobs=-1,
            booster='gbtree',   
            **d
        )    

        metric_rmse, metric_std, best_iteration = my_cv(df, predictors, response, kfolds, xgb, verbose=False)    
        results.append([metric_rmse, metric_std, best_iteration, d])
    
        print("%s %3d result mean: %.6f std: %.6f, iter: %.2f" % (datetime.datetime.strftime(datetime.datetime.now(), "%T"), i, metric_rmse, metric_std, best_iteration))
        
    end_time = datetime.datetime.now()
    print("%-20s %s" % ("Start Time", start_time))
    print("%-20s %s" % ("End Time", end_time))
    print(str(datetime.timedelta(seconds=(end_time-start_time).seconds)))
    
    results_df = pd.DataFrame(results, columns=['rmse', 'std', 'best_iter', 'param_dict']).sort_values('rmse')
    
    best_params = results_df.iloc[0]['param_dict']
    return best_params, results_df