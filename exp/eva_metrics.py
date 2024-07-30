from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np

from utility.configuration import BaseConfig

@dataclass
class EvaluationMetrics(BaseConfig):
    quantile_loss: Dict[str, float]|None = None
    mae: float|None = None
    mse: float|None = None
    rmse: float|None = None
    
    def __post_init__(self):
        if self.quantile_loss is None:
            self.quantile_loss = {}
        if self.mae is None:
            self.mae = -1.0
        if self.mse is None:
            self.mse = -1.0
        if self.rmse is None:
            self.rmse = -1.0

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