import numpy as np


def quantile_loss(pre_values_q, real_values, q):
    """
    give the quantiles and associated value to calculate the quantile loss
    """
    q_loss = np.maximum(q * (real_values - pre_values_q), (q - 1) * (real_values - pre_values_q))
    return q_loss
