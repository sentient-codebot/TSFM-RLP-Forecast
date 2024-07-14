import pandas as pd
import dask.dataframe as dd

class Loaddataset:
    """
    A class to load and preprocess datasets based on resolution and country,
    with an option to split the dataset.
    """ 
    def __init__(self, 
                 resolution: str = None, 
                 country: str = None, 
                 split_ratio: float = 1,
                 random_state: int = 42):
        
        self.resolution = resolution # '60m', '30m', '15m'
        self.country = country # 'uk', 'ge', 'nl', 'aus'
        self.split_ratio = split_ratio # 1 means no split
        self.random_state = random_state # random sample id for train-test split

    def load_dataset(self):
        """
        Load dataset from the path
        """
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
        
        # example : "hf://datasets/Weijie1996/load_timeseries/30m_resolution_ge/ge_30m.parquet"
        path_pre = "hf://datasets/Weijie1996/load_timeseries/"
        path_folder = path_pre + self.resolution + "_resolution_" + self.country + "/"
        path_file = path_folder + self.country + "_" + self.resolution + ".parquet"
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
  
        return train_data, test_data
    

if __name__ == "__main__":
    # check the class
    load = Loaddataset(
        resolution='60m',
        country='nl',
        split_ratio=0.6
    )

    train, test = load.load_dataset()
    print(train.shape, test.shape)
