import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# dataset_path = os.path.join(parent_dir, 'dataset')
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

import dataset.data_loader as dl
import exp.eva_metrics as evm
import utility.configuration as cf

def svr_fit(input, target):
    regrs = []
    for _col in range(target.shape[1]):
        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        regr.fit(input, target[:, _col])
        regrs.append(regr)
    return regrs

def svr_pred(input, regrs):
    # Prepare the output array with zeros, matching the shape (n, d)
    predictions = np.zeros((input.shape[0], len(regrs)))
    
    # Use each trained model to predict its corresponding column
    for i, regr in enumerate(regrs):
        predictions[:, i] = regr.predict(input)
    
    return predictions

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
    split_ratio = 0.3
    
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
                model_name="SVR",
                lookback_window=foo[0].shape[-1],
                prediction_length=foo[1].shape[-1],
            )
            
            # ----------------- Experiment Configuration -----------------
            
            # ----------------- Experiment -----------------
            _mae, _rmse  = [], []
            
            for x , y in tqdm(pair_it, total = len(pair_iterable)//batch_size):
                x = x.reshape(x.shape[0],-1)
                y = y.reshape(y.shape[0],-1)
                
                # fig regressor
                regrs = svr_fit(x[:int(split_ratio*x.shape[0]),:],  y[:int(split_ratio*x.shape[0]),:])
                
                # make predction
                x = x[int(split_ratio*x.shape[0]):,:]
                y = y[int(split_ratio*y.shape[0]):,:]
                y_hat = svr_pred(x, regrs)
                print(x.shape, y.shape, y_hat.shape)
                
                _mae.append(evm.mae(y_hat, y))
                _rmse.append(evm.rmse(y_hat, y))
            
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
            exp_config.append_csv(f'exp/svr/result/{exp_id}.csv')
            # ----------------- Experiment -----------------
            
            # ----------------- Plot the Results-----------
            x = x.reshape(-1, num_steps_day*3)
            plt.plot(range(num_steps_day*3), x[0, :], label='Input', color='b')
            target_range = range(num_steps_day*3, num_steps_day*4)
            plt.plot(target_range, y[0, :], c='r', label='Target')
            plt.plot(target_range, y_hat[0, :], c='g', label='Median')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'SVR Predictions for {country.capitalize()} ({_type.capitalize()})')
            _path = 'exp/svr/result'
            plt.legend()
            plt.savefig(_path + f'/svr_{country}_{reso}_{_type}.png')
            plt.close()
                
                