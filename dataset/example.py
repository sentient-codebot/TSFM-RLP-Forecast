"""define the example dataset provided by GluonTS
>> 'electricity_nips', as in paper: https://arxiv.org/abs/1910.03002
date: 2024-07-01
author: sentient-codebot
"""
from gluonts.dataset.repository import get_dataset, dataset_names

def get_example_dataset(*args, **kwargs):
    return get_dataset("electricity_nips")
