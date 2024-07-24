import numpy as np


def quantile_loss(pre_values_q, real_values, q):
    """
    give the quantiles and associated value to calculate the quantile loss
    """
    q_loss = np.maximum(q * (real_values - pre_values_q), (q - 1) * (real_values - pre_values_q))
    return q_loss

def mae(real_values, pre_values):
    """
    give the real values and the predicted values to calculate the mean absolute error
    """
    mae = np.abs(real_values - pre_values)
    return mae.mean()

def rmse(real_values, pre_values):
    """
    give the real values and the predicted values to calculate the root mean squared error
    """
    rmse = np.sqrt(np.power(real_values - pre_values, 2).mean())
    return rmse.mean()