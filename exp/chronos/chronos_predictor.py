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
            )
            pair_it = dl.array_to_tensor(iter(pair_iterable))
            if reso == '60m':
                pred_length = 24
            elif reso == '30m':
                pred_length = 48
            elif reso == '15m':
                pred_length = 96
            
            _q_10_loss = []
            _q_50_loss = []
            _q_90_loss = []
            _mae_loss = []
            _rmse_loss = []
            pipeline = chronos_prediction()
            
            for x, y in tqdm(pair_it, total=len(pair_iterable)):
                    _input = x
                    _output = y.numpy()
                    forecast = pipeline.predict(_input, pred_length, limit_prediction_length=False)
                    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                    _q_10 = evm.quantile_loss(low, _output, 0.1).mean()
                    _q_50 = evm.quantile_loss(median, _output, 0.5).mean()
                    _q_90 = evm.quantile_loss(high, _output, 0.9).mean()
                    _mae = evm.mae(median, _output)
                    _rmse = evm.rmse(median, _output)
                    _q_10_loss.append(_q_10.item())
                    _q_50_loss.append(_q_50.item())
                    _q_90_loss.append(_q_90.item())
                    _mae_loss.append(_mae.item())
                    _rmse_loss.append(_rmse.item())
                    
            # cancel nan values in the list
            _q_10_loss = [x for x in _q_10_loss if str(x) != 'nan']
            _q_50_loss = [x for x in _q_50_loss if str(x) != 'nan']
            _q_90_loss = [x for x in _q_90_loss if str(x) != 'nan']
            _mae_loss = [x for x in _mae_loss if str(x) != 'nan']
            _rmse_loss = [x for x in _rmse_loss if str(x) != 'nan']
            
            # compute the mean of the loss
            q_10_loss = np.mean(_q_10_loss)
            q_50_loss = np.mean(_q_50_loss)
            q_90_loss = np.mean(_q_90_loss)
            mae_loss = np.mean(_mae_loss)
            rmse_loss = np.mean(_rmse_loss)
            
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print(f"q_10_loss: {q_10_loss}")
            print(f"q_50_loss: {q_50_loss}")
            print(f"q_90_loss: {q_90_loss}")
            print(f"mae_loss: {mae_loss}")
            print(f"rmse_loss: {rmse_loss}")
            
            # plot the prediction
            plt.figure(figsize=(10, 6))
            plt.plot(_output, label='real')
            plt.plot(median, label='prediction')
            plt.fill_between(
                np.arange(len(median)),
                low,
                high,
                alpha=0.3,
                color='red',
                label='uncertainty',
            )
            plt.legend()
            plt.show()
            
            
