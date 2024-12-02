import numpy as np
from sklearn.metrics import mean_squared_error as mse

def RMSPE(y_true, y_pred):
    output = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return output

def RMSE(y_true, y_pred):
    output = np.sqrt(mse(y_true, y_pred))
    return output
