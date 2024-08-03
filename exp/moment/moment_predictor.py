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

from moment_fm import MOMENTPipeline

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
        for _type in ['ind', 'agg']:
            print('--------------------------------------------------')
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print('--------------------------------------------------')
            # load datastet
            pair_iterable = dl.data_for_exp(
                resolution = reso,
                country = country,
                data_type = _type,
                window_split_ratio = 0.75, # TODO 有点混乱
            )
            # pair_iterable.total_pairs = 10 # NOTE only for debug
            pair_it = dl.array_to_tensor(iter(pair_iterable))
            if reso == '60m':
                pred_length = 24
            elif reso == '30m':
                pred_length = 48
            elif reso == '15m':
                pred_length = 96
                
            # ----------------- Experiment Configuration -----------------
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
           
            # Define the pipeline
            model_config = cf.ModelConfig(
                model_name="MOMENT-1-large",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            model = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large", 
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': pred_length,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True, # Freeze the patch embedding layer
                    'freeze_embedder': True, # Freeze the transformer encoder
                    'freeze_head': False, # The linear forecasting head must be trained
                },
            )
            model.init()
            
            # Make prediction
            _input = []
            _output = []
            for x, y in tqdm(pair_it, total=len(pair_iterable)):
                _input.append(x)
                _output.append(y.numpy())
            _input = torch.stack(_input)[:100,:]
            
            # Repeat the last dimension to 512
            _input = _input.repeat(1, 512//_input.shape[-1]+1)[:,:512]
            _input = _input + torch.randn_like(_input)*1e-5
            _output = np.stack(_output)[:100,:] 
            
            # Input is float32              
            _input = _input.float()
            out = model(_input.unsqueeze(0))
            median = out.forecast.squeeze(0)
            
            # Calculate the evaluation metrics
            _mae = evm.mae(median.detach().numpy() , _output)
            _rmse = evm.rmse(median.detach().numpy(), _output)

            # Print the evaluation metrics
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print(f"q_10_loss: {np.nan}")
            print(f"q_50_loss: {np.nan,}")
            print(f"q_90_loss: {np.nan,}")
            print(f"mae_loss: {_mae}")
            print(f"rmse_loss: {_rmse}")
            
            # Output the experiment result
            eval_metrics = evm.EvaluationMetrics(
                quantile_loss={
                    '0.1': np.nan,
                    '0.5': np.nan,
                    '0.9': np.nan,
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
            exp_config.append_csv(f'result/{exp_id}.csv')
            
            # Plot the prediction
            plt.plot(median[0].detach().numpy().flatten(), label='prediction')
            plt.plot(_output[0].flatten(), label='ground truth')
            plt.legend()
            plt.show()
        