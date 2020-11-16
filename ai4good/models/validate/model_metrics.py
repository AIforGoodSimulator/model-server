import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

# MAPE is only available in latest Dev version of sklearn
# the following to be replaced with import when available in stable version of sklearn
#from sklearn.metrics import mean_absolute_percentage_error
def mean_absolute_percentage_error_sklearn(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_errors, weights=multioutput)

def mean_absolute_percentage_error(y_true, y_pred): 
    rows = y_true!=0
    y_true = y_true[rows]
    y_pred = y_pred[rows]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Forecasting Model Metrics
def model_metrics(y,pred):
    metrics= pd.DataFrame(columns=['MAPE','RMSE','MSE', 'MeanAE', 'MedianAE','R2_Score', 'MSLE' ], index=[0])
    metrics['MAPE'] = mean_absolute_percentage_error(y, pred)
    metrics['MeanAE'] = mean_absolute_error(y, pred)
    metrics['MedianAE'] = median_absolute_error(y, pred)
    metrics['MSE'] = mean_squared_error(y, pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['R2_Score'] = r2_score(y, pred)
    metrics['MSLE'] = mean_squared_log_error(y, pred)
    return metrics
