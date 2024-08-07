from typing import Union
import pandas as pd
import dask.dataframe as dd
import numpy as np

class LoadDataset:
    """
    A class to load and preprocess datasets based on resolution and country,
    with an option to split the dataset.
    """
    def __init__(self, 
                 resolution: Union[str, None] = None, 
                 country: Union[str, None] = None, 
                 split_ratio: float = 1,
                 random_state: int = 42):
        
        self.resolution = resolution # '60m', '30m', '15m'
        self.country = country # 'uk', 'ge', 'nl', 'aus'
        self.split_ratio = split_ratio # 1 means no split
        self.random_state = random_state # random sample id for train-test split
    
    def _path(self): 
        path_pre = "hf://datasets/Weijie1996/load_timeseries/"
        path_folder = path_pre + self.resolution + "_resolution_" + self.country + "/"
        path_file = path_folder + self.country + "_" + self.resolution + ".parquet"
        return path_file
    
    def _input_check(self):
        if self.split_ratio < 0 or self.split_ratio > 1:
            raise ValueError("Split ratio should be between 0 and 1")
        if self.resolution is None:
            raise ValueError("Resolution is not set")
        if self.country is None:
            raise ValueError("Country is not set")
        if self.resolution not in ['60m', '30m', '15m']:
            raise ValueError("Resolution should be '60m', '30m', '15m'")
        if self.country not in ['uk', 'ge', 'nl', 'aus']:
            raise ValueError("Country should be 'uk', 'ge', 'nl', 'aus'")
        
    def load_dataset_ind(self):
        """
        Load dataset from the path, the dataset is of individual level.
        """
        self._input_check()
        
        # example : "hf://datasets/Weijie1996/load_timeseries/30m_resolution_ge/ge_30m.parquet"
        path_file = self._path()
        self.data_dd = dd.read_parquet(path_file)
        
        # transform the data to dataframe
        data = self.data_dd.compute()
        
        # sort the data by id and datetime
        data = data.sort_values(by=['id', 'datetime'])
        
        # select the id from the dataset randomly
        id_list = data['id'].unique()
        id_list = pd.Series(id_list).sample(frac=self.split_ratio, random_state=self.random_state)
        
        # split the data into train and test set based on the id
        train_data = data[data['id'].isin(id_list)]
        test_data = data[~data['id'].isin(id_list)]
        
        raise NotImplementedError("Split over ids and time.")
        raise NotImplementedError("make sure split is deterministic.")
  
        return train_data, test_data
    
    def load_dataset_agg(self, 
                         num_agg: int = 2,
                         num_houses: int = 3,
                         random_state: int = 42):
        """
        Load dataset from the path, the dataset is of individual level.
        """
        self._input_check()
        
        # example : "hf://datasets/Weijie1996/load_timeseries/30m_resolution_ge/ge_30m.parquet"
        path_file = self._path()
        self.data_dd = dd.read_parquet(path_file)
        
        # transform the data to dataframe
        data = self.data_dd.compute()
        
        # sort the data by id and datetime
        data = data.sort_values(by=['id', 'datetime'])
        
        # select the id from the dataset randomly
        id_list = data['id'].unique()
        
        # check if len(id_list) is large enough
        if len(id_list) < num_agg:
            raise ValueError("The number of houses is less than the number of aggregation")
        if len(id_list) < num_houses*num_agg:
            raise ValueError("The number of houses is not enough for the aggregation, select a smaller number of houses or number of aggregation")
        
        # randomly select num_houses*agg_num houses in id_list
        id_list = pd.Series(id_list).sample(n=num_houses*num_agg, random_state=random_state)
        
        # split the id_list into num_agg groups and aggregate the data based on the id and  the datetime
        id_list = id_list.reset_index(drop=True)
        id_list = id_list.reset_index()
        id_list['group'] = id_list['index'] // num_houses
        id_list = id_list.drop(columns=['index'])
        id_list = id_list[:num_houses*num_agg]
        
        # merge the data with id_list
        data = data.merge(id_list, left_on='id', right_on=0)
        data['date'] = data['datetime'].dt.strftime('%m-%d')
        data['time'] = data['datetime'].dt.time 
        data = data.drop(columns=['datetime'])
        
        # aggregate the data based on the group and datetime
        data_agg = data.groupby(['group', 'date', 'time']).agg('sum').reset_index()
        
        # sort the data by group and datetime, drop other columns except group, datetime, and the target
        data_agg = data_agg.loc[:, ['group', 'date', 'time', 'target']]
        data_agg = data_agg.sort_values(by=['group', 'date', 'time'])
        data_agg['category'] = self.resolution
        data_agg = data_agg.rename(columns={'group': 'id'})
        
        train_data = data_agg[data_agg['id'] < num_agg*self.split_ratio]
        test_data = data_agg[data_agg['id'] >= num_agg*self.split_ratio]
        
        return train_data, test_data
        
    def _print_information(self):
        print("Resolution: ", self.resolution)
        print("Country: ", self.country)
        print("Split ratio: ", self.split_ratio)
        print("Random state: ", self.random_state)
        
        path_file = self._path()
        self.data_dd = dd.read_parquet(path_file)
        data = self.data_dd.compute()
        data = data.sort_values(by=['id', 'datetime'])
        id_list = data['id'].unique()
        
        print("Number of unique ids: ", len(id_list))
        print("Number of data points: ", data.shape[0])
    

class PairMaker:
    """
    Given a time series dataset, this class generates pairs of time series data for prediction
    """
    def __init__(self, 
                 window_length: int = 100, # the length of the window
                 num_pairs: int = 50, # the number of pairs to generate
                 window_split_ratio: float = 0.5, # the ratio of the window to split
                 random_state: int = 42):
        
        self.window_length = window_length
        self.num_pairs = num_pairs
        self.random_state = random_state
        self.window_split_ratio = window_split_ratio
        
    def _check_input(self):
        if self.window_length < 1:
            raise ValueError("Window length should be larger than 0")
        if self.num_pairs < 1:
            raise ValueError("Number of pairs should be larger than 0")
        if self.window_split_ratio < 0 or self.window_split_ratio > 1:
            raise ValueError("Window split ratio should be between 0 and 1")
    
    def make_pairs(self,
                   data: pd.DataFrame,
                   type_of_split: str = 'overlap' # 'noverlap', 'overlap'
                   ):
        
        # check the input
        self._check_input()

        # check the type of split
        if type_of_split not in ['noverlap', 'overlap']:
            raise ValueError("Type of split should be 'noverlap' or 'overlap'")
        # check the columns of the data, id or group, at least one exists
        if 'id' not in data.columns and 'group' not in data.columns:
            raise ValueError("The data should have 'id' or 'group' column")
        # check the columns of the data, datetime and target, at least one exists
        if 'target' not in data.columns:
            raise ValueError("The data should have 'target' column")
        
        # generate pairs
        pairs = []
        start = 0
        data = data['target']
        for i in range(self.num_pairs):
            if type_of_split == 'overlap':
                # if data.shape[0] < self.window_length+self.num_pairs+1:
                #     raise ValueError("The length of the data is not enough for the window length and number of pairs")
                try:
                    window1 = data.iloc[start:start+ int(self.window_length*self.window_split_ratio)]
                    window2 = data.iloc[start+ int(self.window_length*self.window_split_ratio):start+self.window_length]
                    start += 1
                    pairs.append((np.array(window1), np.array(window2)))
                except IndexError:
                    break
            if type_of_split == 'noverlap':
                # if data.shape[0] < self.window_length*self.num_pairs+1:
                #     raise ValueError("The length of the data is not enough for the window length and number of pairs")
                try:
                    window1 = data.iloc[start:start+ int(self.window_length*self.window_split_ratio)]
                    window2 = data.iloc[start+int(self.window_length*self.window_split_ratio):start+self.window_length]
                    start += self.window_length
                    pairs.append((np.array(window1), np.array(window2)))
                except IndexError:
                    break
        
        # change the pairs to two np.array X and y
        # X = []
        # y = []
        # for pair in pairs:
        #     X.append(pair[0].values)
        #     y.append(pair[1].values)
        
        # X = np.array(X)
        # y = np.array(y)
        
        return pairs

if __name__ == "__main__":
    # Check the class
    load = LoadDataset(
        resolution='60m',
        country='nl',
        split_ratio=0.6
    )

    train, test = load.load_dataset_agg(num_agg=2, num_houses=3)
    print(train.head())
    print(train.shape, test.shape)
    
    pair_maker = PairMaker(
        window_length=20,
        num_pairs=50,
        window_split_ratio=0.5,
        random_state=42
    )
    
    # Select one group of data
    train = train[train['id'] == train['id'].unique()[0]]
    X, Y = pair_maker.make_pairs(train, type_of_split='overlap')
    print(X.shape, Y.shape) 
    
    import matplotlib.pyplot as plt
    _num = 20
    combined = np.concatenate((X[_num], Y[_num]))
    plt.plot(combined)
    plt.plot(X[_num])
    plt.show()
