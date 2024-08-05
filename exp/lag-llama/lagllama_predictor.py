import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf

from lag_llama.gluon.estimator import LagLlamaEstimator

print('--------------------------------------------------')
print(LagLlamaEstimator)
