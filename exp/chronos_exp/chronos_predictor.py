import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)

from typing import Union

from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from chronos import ChronosPipeline

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf
import exp.plot_tool as pt

def chronos_prediction(
    device_map: Union[str, torch.device] = "cpu",
    model_type: str = "amazon/chronos-t5-small",
    torch_dtype: torch.dtype = torch.float32):
  
    # Define the pipeline
    pipeline = ChronosPipeline.from_pretrained(
        model_type,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    return pipeline


if __name__ == "__main__":
    
    # ----------------- Experiment Configuration -----------------
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
        
        for _type in ['ind','agg']:  # 'agg', , 'ind'
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
            batch_size = 48
            pair_it = dl.collate_numpy(pair_iterable, batch_size)
            
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="chronos-t5-small",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            # ----------------- Experiment Configuration -----------------
            
        
            # ----------------- Experiment -----------------
            pipeline = chronos_prediction()
            
            _q_10, _q_50, _q_90, _mae, _rmse  = [], [], [], [], []
            
            for x , y in tqdm(pair_it, total = len(pair_iterable)//batch_size):
                _input = torch.tensor(x).squeeze(1)
                _target = y.squeeze(1)

                forecast = pipeline.predict(_input, num_steps_day, limit_prediction_length=False)

                low, median, high = np.quantile(forecast.numpy(), [0.1, 0.5, 0.9], axis=1)
                low = np.nan_to_num(low, nan=0)
                median = np.nan_to_num(median, nan=0)
                high = np.nan_to_num(high, nan=0)
                
                _q_10.append(evm.quantile_loss(low, _target, 0.1).mean())
                _q_50.append(evm.quantile_loss(median, _target, 0.5).mean())
                _q_90.append(evm.quantile_loss(high, _target, 0.9).mean())
                _mae.append(evm.mae(median, _target))
                _rmse.append(evm.rmse(median, _target))
                
            _q_10, _q_50, _q_90, _mae, _rmse = np.mean(_q_10), np.mean(_q_50), np.mean(_q_90), np.mean(_mae), np.mean(_rmse)

            print('low, median, high shape', low.shape, median.shape, high.shape)
            print('input shape', _input.shape)
            print('target, forecast shape', _target.shape, forecast.shape)
                    
                    
            # eval_metrics = evm.EvaluationMetrics(
            #     quantile_loss={
            #         '0.1': _q_10,
            #         '0.5': _q_50,
            #         '0.9': _q_90,
            #     },
            #     mae=_mae,
            #     rmse=_rmse,
            # )
            
            # print(f"reso: {reso}, country: {country}, type: {_type}")
            # print(f"q_10_loss: {_q_10}")
            # print(f"q_50_loss: {_q_50}")
            # print(f"q_90_loss: {_q_90}")
            # print(f"mae_loss: {_mae}")
            # print(f"rmse_loss: {_rmse}")
            
            # exp_config = cf.ExperimentConfig(
            #     exp_id=exp_id,
            #     data=data_config,
            #     model=model_config,
            #     result=eval_metrics,
            # )
            # exp_config.append_csv(f'/home/wxia/tsfm/TSFM-RLP-Forecast/exp/chronos_exp/result/{exp_id}.csv')
            # ----------------- Experiment-----------------
            
            
            # # # ----------------- Plot the Results-----------
            # plt.plot(_input[0, :], label='Input', color='b')
            
            # # Create the range for the target and predicted values
            # target_range = range(len(_input[0, :]), len(_input[0, :]) + len(_target[0, :]))
            
            # # Plot the target sequence
            # plt.plot(target_range, _target[0, :], c='r', label='Target')
            
            # # Plot the median prediction
            # plt.plot(target_range, median[0, :], c='g', label='Median')
            
            # # Fill the area between low and high predictions
            # plt.fill_between(target_range, low[0, :], high[0, :], color='gray', alpha=0.3, label='Uncertainty')
            
            # # Set plot labels and title
            # plt.xlabel('Time')
            # plt.ylabel('Value')
            # plt.title(f'Chronos Predictions for {country.capitalize()} ({_type.capitalize()})')
            
            # # Add a legend
            # # plt.legend()

            # # Save the plot
            # _path = 'exp/chronos_exp/result/'
            # plt.savefig(_path + f'chronos_{country}_{reso}_{_type}.png')
            # plt.close()
        
            # ----------------- Plot the Results-----------
            # _path = 'exp/chronos_exp/result/'
            pt.plot_chronos_predictions(_input, _target, median, low, high, country, reso, _type, _path = 'exp/chronos_exp/result/')

            
            
            
