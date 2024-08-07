import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from nixtla import NixtlaClient

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf


def numpy_to_dataframe(x, freq='15T'):
    x = pd.DataFrame(x)
    x['unique_id'] = x[0]
    x['value'] = x[1]   
    x = x.drop(columns=[0])
    x = x.drop(columns=[1])
    x['timestamp'] = pd.date_range(start='2020-01-01', periods=len(x), freq=freq)
    return x
    

if __name__ == "__main__":
    # init nixtla client
    nixtla_client = NixtlaClient(
    api_key = 'nixtla-tok-vexIpfKzF7DmJ1YJ8UKuLVb6kq21AJBGSDsokQzYUt9RjipYXmOfZUz1ZOBGFdp4etXrjKHDSRQuT3rI'
    )

    reso_country = [
            ('60m', 'nl'),
            # ('60m', 'ge'),
            # ('30m', 'ge'),
            # ('15m', 'ge'),
            # ('30m', 'uk'),
            # ('60m', 'uk'),
        ]
    
    # -------- Experiment Configuration --------
    exp_id = cf.generate_time_id()
    
    for reso, country in reso_country:
        for _type in ['ind', 'agg']:
            print('--------------------------------------------------')
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print('--------------------------------------------------')
            
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
                freq = '60T'
            elif reso == '30m':
                pred_length = 48
                freq = '30T'
            elif reso == '15m':
                pred_length = 96
                freq = '15T' 
            
            # data config
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))   
            model_config = cf.ModelConfig(
                model_name="timegpt",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            # start the data
            _input = []
            _output = []
            for x, y in tqdm(pair_it, total=len(pair_iterable)):
                _input.append(x)
                _output.append(y.numpy())
            _input = torch.stack(_input)
            _output = np.stack(_output) 
            
            # fill nan with 0
            _input = torch.nan_to_num(_input)
            _output = np.nan_to_num(_output)
            
            # process to from parallel 
            _id = torch.tensor(range(len(_input)))
            _id = _id.reshape(-1, 1).repeat(1, _input.shape[1])
            _id = _id.reshape(-1, 1)
            _input = _input.reshape(-1,1)
            _input = torch.cat(( _id, _input), 1).numpy()
            
            _input = numpy_to_dataframe(_input, freq=freq)
            
   
            # make prediction
            forecast_df = nixtla_client.forecast(
                df=_input,
                h=pred_length,
                time_col='timestamp',
                target_col="value",
                freq=freq
            )  
            
            # evaluate the model
            y_hat = forecast_df['TimeGPT'].values
            y_hat = y_hat.reshape(-1, pred_length)
            _mae = evm.mae(y_hat, _output)
            _rmse = evm.rmse(y_hat, _output)
            
            eval_metrics = evm.EvaluationMetrics(
                quantile_loss={
                    '0.1': 0,
                    '0.5': 0,
                    '0.9': 0,
                },
                mae=_mae,
                rmse=_rmse,
            )
            
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print(f"mae_loss: {_mae}")
            print(f"rmse_loss: {_rmse}")
            
            exp_config = cf.ExperimentConfig(
                exp_id=exp_id,
                data=data_config,
                model=model_config,
                result=eval_metrics,
            )
            exp_config.append_csv(f'result/{exp_id}.csv')
            
            # Plot predictions
            plt.figure(figsize=(12, 6))
            plt.plot(y, label='Actual')
            plt.plot(forecast_df['TimeGPT'].iloc[-24:].values, label='Forecast')
            plt.legend()
            plt.show()
            break

