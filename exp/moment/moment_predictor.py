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
from pprint import pprint

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf

from momentfm import MOMENTPipeline
import exp.plot_tool as pt

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
        
        for _type in ['ind', 'agg']:  # 'agg', , 'ind'
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
            pair_it = dl.collate_numpy(pair_iterable, batch_size)
            
                
            # ----------------- Experiment Configuration -----------------
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="AutonLab/MOMENT-1-large",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            # ----------------- Experiment Configuration -----------------
            
        
            # ----------------- Experiment -----------------
            model = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large", 
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': num_steps_day,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True, # Freeze the patch embedding layer
                    'freeze_embedder': True, # Freeze the transformer encoder
                    'freeze_head': False, # The linear forecasting head must be trained
                },
            )
            model.init()
            
            # Make predictionpi
            _mae, _rmse  = [], []
            
            for x , y in tqdm(pair_it, total = len(pair_iterable)//batch_size):
                _input = torch.tensor(x).float()
                _input = _input.repeat(1, 512//x.shape[2]+1, 1).reshape(x.shape[0], 1, -1)
                _input = _input[:,:,-512:]
                _target = y.squeeze(1)
                out = model(_input).forecast.squeeze(0)
                
                print(out.shape, num_steps_day, _target.shape)
                _mae.append(evm.mae(out.detach().numpy(), _target))
                _rmse.append(evm.rmse(out.detach().numpy() , _target))

            _mae, _rmse = np.mean(_mae), np.mean(_rmse)
            # Output the experiment result
            eval_metrics = evm.EvaluationMetrics(
                quantile_loss={
                    '0.1': -1,
                    '0.5': -1,
                    '0.9': -1,
                },
                mae=_mae,
                rmse=_rmse,
            )
                        
            exp_config = cf.ExperimentConfig(
                exp_id=exp_id,
                data=data_config,
                model=model_config,
                result=eval_metrics,
            )
            exp_config.append_csv(f'exp/moment/result/{exp_id}.csv')
            
            # ----------------- Plot the Results-----------
            # print(x.shape, _target[0, :].shape, out[0, 0, :].detach().numpy().shape)
            # plt.plot(range(x.shape[2]), x[0, 0, :], label='Input', color='b')
            # target_range = range(x.shape[2], x.shape[2] + len(_target[0, :]))
            # plt.plot(target_range, _target[0, :], c='r', label='Target')
            # plt.plot(target_range, out[0, 0, :].detach().numpy(), c='g', label='Median')
            # plt.xlabel('Time')
            # plt.ylabel('Value')
            # plt.title(f'Chronos Predictions for {country.capitalize()} ({_type.capitalize()})')
            # _path = 'exp/moment/result/'
            # plt.savefig(_path + f'moment_{country}_{reso}_{_type}.png')
            # plt.close()
            
            _path = 'exp/moment/result'
            print(x.shape, y.shape)
            x = x[:, 0, :]
            y = _target
            y_hat = out[:, 0, :].detach().numpy()
            print(x.shape, y.shape, y_hat.shape)
            pt.plot_predictions_point(x, y, y_hat, country, reso, _type, 'moment', num_steps_day, _path)
              
            
                    
            