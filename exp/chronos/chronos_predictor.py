import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dataset_path = os.path.join(parent_dir, 'dataset')
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
            # load datastet
            x, y = dl.data_for_exp(
                resolution = reso,
                country = country,
                data_type = _type,
            )
            
            _q_10_loss = []
            _q_50_loss = []
            _q_90_loss = []
            _mse_loss = []
            pipline = chronos_prediction()
            
            for i in tqdm(range(x.shape[0])):
                for j in range(1):  # x.shape[1]
                    _input = x[i, j, :]
                    _output = y[i, j, :]  
                    forecast = pipline.predict(_input, 64)
                    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                    _q_10 = evm.quantile_loss(low, _output, 0.1).mean()
                    _q_50 = evm.quantile_loss(median, _output, 0.5).mean()
                    _q_90 = evm.quantile_loss(high, _output, 0.9).mean()
                    _mse = ((_output- median)**2).mean()
                    _q_10_loss.append(_q_10.item())
                    _q_50_loss.append(_q_50.item())
                    _q_90_loss.append(_q_90.item())
                    _mse_loss.append(_mse.item())
                    
            # cancel nan values in the list
            _q_10_loss = [x for x in _q_10_loss if str(x) != 'nan']
            _q_50_loss = [x for x in _q_50_loss if str(x) != 'nan']
            _q_90_loss = [x for x in _q_90_loss if str(x) != 'nan']
            _mse_loss = [x for x in _mse_loss if str(x) != 'nan']
            
            # compute the mean of the loss
            q_10_loss = np.mean(_q_10_loss)
            q_50_loss = np.mean(_q_50_loss)
            q_90_loss = np.mean(_q_90_loss)
            mse_loss = np.mean(_mse_loss)
            
            print(f"reso: {reso}, country: {country}, type: {_type}")
            print(f"q_10_loss: {q_10_loss}")
            print(f"q_50_loss: {q_50_loss}")
            print(f"q_90_loss: {q_90_loss}")
            print(f"mse_loss: {mse_loss}")
            
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
            
            
