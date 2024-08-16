import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)
from typing import List

import matplotlib.pyplot as plt
import numpy as np
# import torch
from tqdm import tqdm
import timesfm

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf

def pad_sequence(
    sequence: List,
    target_length,
):
    if len(sequence) >= target_length:
        return sequence[-target_length:]
    else:
        return [0.] * (target_length - len(sequence)) + sequence

def get_timesfm_predictor(
    context_length: int=96,
    prediction_length: int=24,
):
    assert context_length % 32 == 0
    timesfm_backend = 'gpu'
    from jax._src import config
    config.update(
        "jax_platforms", {"cpu": "cpu", "gpu": "cuda", "tpu": ""}[timesfm_backend]
    )
    model = timesfm.TimesFm(
        context_len=context_length,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend=timesfm_backend,
    )
    model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    
    return model

if __name__ == "__main__":
    reso_country = [
            ('60m', 'nl'),
            ('60m', 'ge'),
            ('30m', 'ge'),
            ('15m', 'ge'),
            ('30m', 'uk'),
            ('60m', 'uk'),
        ]
    
    # -------- Experiment Configuration --------
    exp_id = cf.generate_time_id()
    
    for reso, country in reso_country:
        if reso == '60m':
            num_steps_day = 24
        elif reso == '30m':
            num_steps_day = 48
        elif reso == '15m':
            num_steps_day = 96
        for _type in ['ind', 'agg']:
            print('--------------------------------------------------')
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print('--------------------------------------------------')
            # load datastet
            pair_iterable = dl.data_for_exp(
                resolution = reso,
                country = country,
                data_type = _type,
                context_length=num_steps_day*3,
                prediction_length=num_steps_day,
            )
            # pair_iterable.total_pairs = 10 # NOTE only for debug
            batch_size = 128
            pair_it = dl.collate_list(dl.array_to_list(iter(pair_iterable)), batch_size=batch_size)
            
            # ----------------- Experiment Configuration -----------------
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="timesfm-200m",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            model = get_timesfm_predictor(
                context_length=96,
                prediction_length=num_steps_day,
            )
            
            _input = []
            _target = []
            _forecast = []
            for x, y in tqdm(pair_it, total=len(pair_iterable)//batch_size):
                _input += x
                _target += y
                y_pred = model.forecast(
                    inputs=x,
                    freq=[0]*len(x),
                )
                _forecast += y_pred
            _target = np.concatenate(_target, axis=0)
            forecast = np.concatenate(_forecast, axis=0)
            print(forecast.shape)
            
            _mae = evm.mae(forecast, _target)
            _rmse = evm.rmse(forecast, _target)
            _q_10 = -1.0
            _q_50 = -1.0
            _q_90 = -1.0
                    
            eval_metrics = evm.EvaluationMetrics(
                quantile_loss={
                    '0.1': _q_10,
                    '0.5': _q_50,
                    '0.9': _q_90,
                },
                mae=_mae,
                rmse=_rmse,
            )
            
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print(f"q_10_loss: {_q_10}")
            print(f"q_50_loss: {_q_50}")
            print(f"q_90_loss: {_q_90}")
            print(f"mae_loss: {_mae}")
            print(f"rmse_loss: {_rmse}")
            
            exp_config = cf.ExperimentConfig(
                exp_id=exp_id,
                data=data_config,
                model=model_config,
                result=eval_metrics,
            )
            exp_config.append_csv(f'result/{exp_id}.csv')
            
    print('complete.')