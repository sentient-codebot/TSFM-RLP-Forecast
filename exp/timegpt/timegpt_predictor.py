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
    
    # ----------------- Experiment Configuration -----------------
    # init nixtla client
    nixtla_client = NixtlaClient(
    api_key = 'nixtla-tok-vexIpfKzF7DmJ1YJ8UKuLVb6kq21AJBGSDsokQzYUt9RjipYXmOfZUz1ZOBGFdp4etXrjKHDSRQuT3rI'
    )
    
    reso_country = [
        ('60m', 'nl'),
        ('60m', 'ge'),
        ('30m', 'ge'),
        ('15m', 'ge'),
        ('30m', 'uk'),
        ('60m', 'uk'),
    ]

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
            
            # ----------------- Experiment Configuration -----------------
            
            # ----------------- Experiment -----------------
            freq = reso + 'in'
            _mae, _rmse  = [], []
            # start the data
            for x, y in tqdm(pair_it, total=len(pair_iterable)//batch_size):
                x = x.squeeze(1)
                y = y.squeeze(1)
                id = torch.tensor(np.arange(1, x.shape[0]+1)).view(-1, 1)
                id = id.repeat(1, x.shape[1]).view(-1, 1).numpy()
                date = pd.date_range(start='2020-01-01', periods=x.shape[1], freq=freq)
                # repeat date for each id
                date = np.tile(date, x.shape[0])
                x = x.reshape(-1, 1)
                df = pd.DataFrame(np.hstack((id,  x)), columns=['unique_id', 'y'])
                _df = pd.concat([df, pd.Series(date, name='ds')], axis=1)

                # make prediction
                forecast_df = nixtla_client.forecast(
                    df=_df ,
                    h=y.shape[1],
                )  
                
                # evaluate the model
                y_hat = forecast_df['TimeGPT'].values
                y_hat =  y_hat.reshape(-1, y.shape[1])
            
                _mae.append(evm.mae(y_hat, y))
                _rmse.append(evm.rmse(y_hat, y))
            
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
            exp_config.append_csv(f'exp/timegpt/result/{exp_id}.csv')
            # ----------------- Experiment -----------------
            
            # ----------------- Plot the Results-----------
            x = x.reshape(-1, num_steps_day*3)
            plt.plot(range(num_steps_day*3), x[0, :], label='Input', color='b')
            target_range = range(num_steps_day*3, num_steps_day*4)
            plt.plot(target_range, y[0, :], c='r', label='Target')
            plt.plot(target_range, y_hat[0, :], c='g', label='Median')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Chronos Predictions for {country.capitalize()} ({_type.capitalize()})')
            _path = 'exp/timegpt/result'
            plt.legend()
            plt.savefig(_path + f'/timegpt_{country}_{reso}_{_type}.png')

            plt.close()
