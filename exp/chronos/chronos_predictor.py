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
from chronos import ChronosPipeline

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf


def chronos_prediction(
    device_map: Union[str, torch.device] = "cpu",
    model_type: str = "amazon/chronos-t5-tiny",
    torch_dtype: torch.dtype = torch.float32):
  
    # Define the pipeline
    pipeline = ChronosPipeline.from_pretrained(
        model_type,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    return pipeline


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
            pair_it = dl.array_to_tensor(iter(pair_iterable))
            
            # ----------------- Experiment Configuration -----------------
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="chronos-t5-tiny",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            pipeline = chronos_prediction()
            
            _input = []
            _target = []
            for x, y in tqdm(pair_it, total=len(pair_iterable)):
                _input.append(x)
                _target.append(y.numpy())
            _input = torch.stack(_input)
            _target = np.stack(_target)
            forecast = pipeline.predict(_input, num_steps_day, limit_prediction_length=False)
            print('input shape', _input.shape)
            print('target, forecast shape', _target.shape, forecast.shape)
            low, median, high = np.quantile(forecast.numpy(), [0.1, 0.5, 0.9], axis=1)
            print('low, median, high shape', low.shape, median.shape, high.shape)

            # report the nan with 0
            low = np.nan_to_num(low, nan=0)
            median = np.nan_to_num(median, nan=0)
            high = np.nan_to_num(high, nan=0)
            
            _q_10 = evm.quantile_loss(low, _target, 0.1).mean()
            _q_50 = evm.quantile_loss(median, _target, 0.5).mean()
            _q_90 = evm.quantile_loss(high, _target, 0.9).mean()
            _mae = evm.mae(median, _target)
            _rmse = evm.rmse(median, _target)
                    
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
            
        
            