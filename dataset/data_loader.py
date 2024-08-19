import os 
import sys
from typing import Dict
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# import torch
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm
import yaml

import dataset.data_process as dp

class PairIterable:
    def __init__(
        self,
        df: pd.DataFrame,
        prediction_length: int = 24,
        context_length: int = 72,
        # num_houses: int,
        # num_pairs: int = 60,
        total_pairs: int = 1800,
        random_state: int = 0000,
        # window_length: int,
        # window_split_ratio: float,
    ):
        pair_maker = dp.PairMaker(
            window_length=prediction_length+context_length,
            window_split_ratio=(context_length)/float(context_length+prediction_length),
            random_state=random_state
        )
        self.all_ids = df['id'].unique()
        self.id_pairs = {}
        for id in self.all_ids:
            _df = df[df['id'] == id]
            pairs = pair_maker.make_pairs(_df, 'noverlap')
            self.id_pairs[id] = pairs
        self.total_pairs = total_pairs
        print(f"Total pairs used: {self.total_pairs}")
        print(f"Total available pairs: {self.total_available_pairs}")
        if self.total_available_pairs < self.total_pairs:
            print(f"Warning: total available pairs is {self.total_available_pairs}, less than total pairs required {self.total_pairs}")
            
    @property
    def total_available_pairs(self):
        return sum([len(self.id_pairs[id]) for id in self.all_ids])
            
    def __len__(self):
        return self.total_pairs
            
    def __iter__(self):
        self.__gen = self.__create_generator()
        return self
    
    def __create_generator(self):
        _pair_count = 0
        for id in self.all_ids:
            if len(self.id_pairs[id]) == 0:
                continue
            for pair in self.id_pairs[id]:
                _pair_count += 1
                yield pair
                if _pair_count == self.total_pairs:
                    return # instead of raise StopIteration
                    # raise StopIteration # no more pairs
        # reach here if no enough pairs or just enough pairs
    
    def __next__(self):
        return next(self.__gen)
    
class LazyPairIterable:
    def __init__(
        self,
        df: dd.DataFrame,
        prediction_length: int = 24,
        context_length: int = 72,
        total_pairs: int = 1800,
        random_state: int = 0000,
    ):
        self.pair_maker = dp.PairMaker(
            window_length=prediction_length+context_length,
            window_split_ratio=(context_length)/float(context_length+prediction_length),
            random_state=random_state
        )
        # NOTE
        #   df's partitions were re-indexed. 
        self.df = df
        self.total_pairs = total_pairs
        print('Total pairs specified:', self.total_pairs)
        
    def __len__(self):
        return self.total_pairs
    
    def __iter__(self):
        self.__gen = self.__create_generator()
        return self
    
    def __create_generator(self):
        _pair_count = 0
        for idx_row_group in range(self.df.npartitions):
            # make pairs
            row_group = self.df.get_partition(idx_row_group).compute()
            pairs = self.pair_maker.make_pairs(
                row_group,
                'noverlap'
            )
            for pair in pairs:
                _pair_count += 1
                yield pair
                if _pair_count == self.total_pairs:
                    return
                
    def __next__(self):
        return next(self.__gen)

def batch_generator(pair_iterable, batch_size=120):
    X_batch = []
    y_batch = []
    
    iterator = iter(pair_iterable)
    
    for x, y in iterator:
        df = pd.DataFrame(np.hstack([x.reshape(1, -1), y.reshape(1, -1)]))
        
        df.dropna(inplace=True)
        
        if not df.empty:
            x_clean, y_clean = df.iloc[:, 0].values, df.iloc[:, 1].values
            X_batch.append(x_clean)
            y_batch.append(y_clean)
        
        if len(X_batch) == batch_size:
            yield np.array(X_batch), np.array(y_batch)
            X_batch = []
            y_batch = []
            
    if X_batch:
        yield np.array(X_batch), np.array(y_batch)


def array_to_list(it):
    for x, y in it:
        yield x.tolist(), y.tolist()
            
def array_to_tensor(it):
    import torch
    for x, y in it:
        yield torch.tensor(x), torch.tensor(y)
        
def collate_list(it, batch_size):
    "collate list into list"
    while True:
        list_x = []
        list_y = []
        while len(list_x) < batch_size:
            try:
                x, y = next(it)
                list_x.append(x)
                list_y.append(y)
            except StopIteration:
                break
        if len(list_x) == 0:
            return
        yield list_x, list_y
        
def collate_fn(it, batch_size):
    import torch
    while True:
        list_x = []
        list_y = []
        while len(list_x) < batch_size:
            try:
                x, y = next(it)
                list_x.append(x)
                list_y.append(y)
            except StopIteration:
                break
        if len(list_x) == 0:
            return
        yield torch.stack(list_x, dim=0), torch.stack(list_y, dim=0)

def data_for_exp(
        resolution: str = '60m',
        country: str = 'nl',
        data_type: str = 'ind',
        prediction_length: int = 24,
        context_length: int = 72,
        window_split_ratio: float = 0.75, 
        random_state: int = 42
        ):
        # load the YAML file
        with open("dataset/data_loader_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # raise NotImplementedError("window_split_ratio -> int")
        if country == 'uk':
            loader = dp.LoadUKDataset(
                resolution=resolution,
                country=country,
                split_ratio=config['uk_split_ratio']
            )
        else:
            loader = dp.LoadDataset(
                    resolution=resolution,
                    country=country,
                    split_ratio=config['uk_split_ratio']
                )

        reso_country = [
            ('60m', 'nl'),
            ('60m', 'ge'),
            ('30m', 'ge'),
            ('15m', 'ge'),
            ('30m', 'uk'),
            ('60m', 'uk'),
        ]
        
        # check if resolution, country is in reso_country
        for reso, cnt in reso_country:
            if reso == resolution and cnt == country:
                found = True
                break

        if not found:
            raise ValueError("Resolution or country not found in reso_country")
        # write case for each tuple in reso_country
        
        if resolution == '60m' and country == 'nl':
            if data_type == 'ind':
                houses = config['nl_60']['houses_ind']
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['nl_60']['total_pairs_ind'],
                    random_state=random_state
                )   
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = config['nl_60']['num_agg'],
                    num_houses = config['nl_60']['num_houses_agg'],
                    random_state = random_state
                )
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['nl_60']['total_pairs_agg'],
                    random_state=random_state
                )
        elif resolution == '60m' and country == 'ge':
            if data_type == 'ind':
                houses = config['ge_60']['houses_ind']
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_60']['total_pairs_ind'],
                    random_state=random_state
                )
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = config['ge_60']['num_agg'],
                    num_houses = config['ge_60']['num_houses_agg'],
                    random_state = random_state
                )
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_60']['total_pairs_agg'],
                    random_state=random_state
                )  
        elif resolution == '30m' and country == 'ge':
            if data_type == 'ind':
                houses = config['ge_30']['houses_ind']
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_30']['total_pairs_ind'],
                    random_state=random_state
                )
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = config['ge_30']['num_agg'],
                    num_houses = config['ge_30']['num_houses_agg'],
                    random_state = random_state
                )
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_30']['total_pairs_agg'],
                    random_state=random_state
                )  
        elif resolution == '15m' and country == 'ge':
            if data_type == 'ind':
                houses = config['ge_15']['houses_ind']
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_15']['total_pairs_ind'],
                    random_state=random_state
                )
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = config['ge_15']['num_agg'],
                    num_houses = config['ge_15']['num_houses_agg'],
                    random_state = random_state
                )
                print("Making pairs")
                pair_it = PairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['ge_15']['total_pairs_agg'],
                    random_state=random_state
                )
        elif resolution == '30m' and country == 'uk':
            if data_type == 'ind':
                print("Loading individual data")
                df_train, df_test = loader.load_dataset_ind()
                print("Making pairs")
                pair_it = LazyPairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['uk_30']['total_pairs_ind'],
                    random_state=random_state
                )
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_houses = config['uk_30']['num_houses_agg'],
                )
                print("Making pairs")
                pair_it = LazyPairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['uk_30']['total_pairs_agg'],
                    random_state=random_state
                )
        elif resolution == '60m' and country == 'uk':
            if data_type == 'ind':
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                print("Making pairs")
                pair_it = LazyPairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['uk_60']['total_pairs_ind'],
                    random_state=random_state
                )            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_houses = config['uk_60']['num_houses_agg'], # better < number of ids in each partition
                )
                print("Making pairs")
                pair_it = LazyPairIterable(
                    df_train,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    total_pairs=config['uk_60']['total_pairs_agg'],
                    random_state=random_state
                )
        else:
            raise ValueError("Invalid resolution or country")

        return pair_it

                        
if __name__ == "__main__":
    
    #    reso_country = [
    #         ('60m', 'nl'),
    #         ('60m', 'ge'),
    #         ('30m', 'ge'),
    #         ('15m', 'ge'),
    #         ('30m', 'uk'),
    #         ('60m', 'uk'),
    #     ]
    
    # data type ind, agg
    pair_it = data_for_exp(
        resolution ='15m',
        country = 'ge',
        data_type = 'agg',
    )
    print('Length:', pair_it.__len__())
    x, y = next(iter(pair_it))
    print(x.shape, y.shape)