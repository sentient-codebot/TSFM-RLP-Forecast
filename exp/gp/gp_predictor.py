import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf
import exp.plot_tool as pt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def train_gp_models(X, y):
    models = []
    for i in range(y.shape[1]):
        # Define the kernel for the Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # Initialize and train the Gaussian Process model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y[:, i])
        models.append(gp)
    return models

def gp_predict_quantiles(models, X_test):
    means = np.zeros((X_test.shape[0], len(models)))
    stds = np.zeros((X_test.shape[0], len(models)))
    
    # Obtain mean and standard deviation predictions for each output dimension
    for i, model in enumerate(models):
        mean, std = model.predict(X_test, return_std=True)
        means[:, i] = mean
        stds[:, i] = std
    
    # Calculate 10% and 90% quantiles using the mean and standard deviation
    # quantiles = {
    #     '10%': means - 1.28 * stds,  # 10% quantile = mean - 1.28 * std (for 10% quantile)
    #     'mean': means,
    #     '90%': means + 1.28 * stds   # 90% quantile = mean + 1.28 * std (for 90% quantile)
    # }
    
    return means - 1.28 * stds, means,  means + 1.28 * stds


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
    split_ratio = 0.6
    
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
            batch_size = 128*2
            pair_it = dl.collate_numpy(pair_iterable, batch_size)
            
            data_config = cf.DataConfig(
                country=country,
                resolution=reso,
                aggregation_type=_type,
            )
            foo = next(iter(pair_iterable))
            model_config = cf.ModelConfig(
                model_name="GP",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            # ----------------- Experiment Configuration -----------------
            
            # ----------------- Experiment -----------------
            _q_10, _q_50, _q_90, _mae, _rmse  = [], [], [], [], []
            
            for x , y in tqdm(pair_it, total = len(pair_iterable)//batch_size):
                x = x.reshape(x.shape[0],-1)
                y = y.reshape(y.shape[0],-1)
                
                # fig regressor
                regrs = train_gp_models(x[:int(split_ratio*x.shape[0]),:],  y[:int(split_ratio*x.shape[0]),:])
                
                # make predction
                x = x[int(split_ratio*x.shape[0]):,:]
                y = y[int(split_ratio*y.shape[0]):,:]
                low, mean, high = gp_predict_quantiles(regrs, x)
                ow = np.nan_to_num(low, nan=0)
                mean = np.nan_to_num(mean, nan=0)
                high = np.nan_to_num(high, nan=0)
                
                _q_10.append(evm.quantile_loss(low, y, 0.1).mean())
                _q_50.append(evm.quantile_loss(mean, y, 0.5).mean())
                _q_90.append(evm.quantile_loss(high, y, 0.9).mean())
                _mae.append(evm.mae(mean, y))
                _rmse.append(evm.rmse(mean, y))
                
            _q_10, _q_50, _q_90, _mae, _rmse = np.mean(_q_10), np.mean(_q_50), np.mean(_q_90), np.mean(_mae), np.mean(_rmse)

            print('low, median, high shape', low.shape, mean.shape, high.shape)
            print('input shape', x.shape)
            print('target, forecast shape', y.shape, mean.shape)
                    
                    
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
            exp_config.append_csv(f'exp/gp/result/{exp_id}.csv')
            # ----------------- Experiment -----------------
            
            # ----------------- Plot the Results-----------
            pt.plot_gp_predictions(x, y, mean, low, high, country, reso, _type,  _path = 'exp/gp/result/')
