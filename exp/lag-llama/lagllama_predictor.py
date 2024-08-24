import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)
from typing import Union
import pandas as pd

import matplotlib.pyplot as plt
from itertools import islice
import matplotlib.dates as mdates
import numpy as np
import torch
from tqdm import tqdm

from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset
import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf

from lag_llama.gluon.estimator import LagLlamaEstimator

ckpt_path = os.path.abspath(os.path.join(parent_dir, './lag-llama', './lag-llama.ckpt'))

# debug = os.getenv('DEBUG', 'False') == 'True'

def predict(dataset, prediction_length: int, context_length=32, batch_size=128, use_rope_scaling=False, num_samples=100):
    """A function for Lag-Llama inference.
    Copy from https://colab.research.google.com/drive/1XxrLW9VGPlZDw3efTvUi0hQimgJOwQG6#scrollTo=gyH5Xq9eSvzq&line=1&uniqifier=1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        num_parallel_samples=100, # TODO: maybe change to num_samples
        device=device,
        batch_size=batch_size,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss

def create_pd_dataset(data, freq):
    """
    Convert a (m, n) numpy array into a combined pandas DataFrame for multiple units (e.g., families).

    Parameters:
    data (np.ndarray): Input numpy array with shape (m, n).
    freq (str): Frequency of time intervals, default is 'H' (hourly).

    Returns:
    pd.DataFrame: Combined DataFrame containing 'item_id', 'timestep', and 'target' columns.
    """
    num_units, num_intervals = data.shape

    df_list = []

    date_range = pd.date_range(start='2000-01-01', periods=num_intervals, freq=freq)

    # Loop over each unit (e.g., family) to create individual DataFrames
    for unit_id in range(num_units):
        # Create a DataFrame for the current unit
        df = pd.DataFrame({
            'item_id': unit_id,
            'target': data[unit_id]
        } , index=date_range)

        # Set numerical columns as float32
        for col in df.columns:
            # Check if column is not of string type
            if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
                df[col] = df[col].astype('float32')
        df_list.append(df)

    return PandasDataset(df_list, target="target", freq=freq)

def generate_dataset(x, y, freq):
    """
    Generate a dataset for lag-llama predictor without progress tracking.

    Args:
        x: input data
        y: output data
        freq (str): The frequency of the data

    Returns:
        tuple: A tuple containing the PandasDataset dataset, original dataset, and output.
    """
    # lag-llama need to include the timesteps in the dataframe that we want to perform prediction
    # so we fill the timesteps with dummy values
    combined = np.hstack((x, np.zeros_like(y)))
    # combined = np.hstack((_input, _output)) # test the impact of including the target instead of zeros
    input_dataset = create_pd_dataset(combined, freq)

    return input_dataset

if __name__ == "__main__":
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
            if reso == '60m':
                num_steps_day = 24
                freq = '60min'
            elif reso == '30m':
                num_steps_day = 48
                freq = '30min'
            elif reso == '15m':
                num_steps_day = 96
                freq = '15min'

            # load datastet
            pair_iterable = dl.data_for_exp(
                resolution = reso,
                country = country,
                data_type = _type,
                context_length=num_steps_day*3,
                prediction_length=num_steps_day,
            )
            # pair_iterable.total_pairs = 10 # NOTE only for debug
            batch_size = 3
            pair_it = dl.collate_np(dl.filter_nan(iter(pair_iterable)), batch_size)

            # ----------------- Experiment Configuration -----------------
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="lag-llama",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )

            _target = []
            _input = []
            forecasts_result = []
            forecast_quantiles = []
            for x, y in tqdm(pair_it, total=len(pair_iterable)//batch_size):
                _target.extend(y)
                _input.extend(x)
                input_dataset = generate_dataset(x, y, freq)
                # make prediction
                forecasts, tss = predict(input_dataset, num_steps_day, batch_size=batch_size)
                for forecast in forecasts:
                    # we can get median by `forecasts[i].median` as well
                    forecast_quantiles.append(np.quantile(forecast.samples, [0.1, 0.5, 0.9], axis=0))
                    forecasts_result.append(forecast.samples)

            _input = np.array(_input)
            _target = np.array(_target)
            forecasts_result = np.array(forecasts_result)
            forecast_quantiles = np.array(forecast_quantiles)
            low, median, high = forecast_quantiles[:, 0, :], forecast_quantiles[:, 1, :], forecast_quantiles[:, 2, :]

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

            # # Plot first 9 units' predictions
            # plt.figure(figsize=(20, 15))
            # date_formater = mdates.DateFormatter('%b, %d')
            # plt.rcParams.update({'font.size': 15})

            # # Iterate through the first 9 series, and plot the predicted samples
            # for idx, target in enumerate(_target):
            #     if idx >= 9:
            #         break
            #     ax = plt.subplot(3, 3, idx + 1)
            #     previous = _input[idx]
            #     forecast = forecasts_result[idx]
            #     x = np.append(previous, target)
            #     ax.plot(range(len(x)), x, label='Actual', color='b', linewidth=2)
            #     ax.plot(range(len(previous), len(x)), median[idx], label='Median Forecast', color='g', linewidth=2)

            #     ml, mh = np.quantile(forecast, [0.25, 0.75], axis=0)
            #     ax.fill_between(range(len(previous), len(x)), low[idx], high[idx], color='g', alpha=0.2)
            #     ax.fill_between(range(len(previous), len(x)), ml, mh, color='g', alpha=0.4)

            #     ax.set_title(f'Series {idx + 1}')
            #     ax.xaxis.set_major_formatter(date_formater)
            #     ax.set_xlabel('Date')
            #     ax.set_ylabel('Value')

            # plt.gcf().tight_layout()
            # plt.legend()
            # plt.show()
