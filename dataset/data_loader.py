import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import dataset.data_process as dp
from tqdm import tqdm


def data_for_exp(
        resolution: str = '60m',
        country: str = 'nl',
        data_type: str = 'ind',
        prediction_length: int = 24,
        window_split_rato: float = 0.75,
        random_state: int = 42
    ):
        loader = dp.LoadDataset(
                resolution=resolution,
                country=country,
                split_ratio=1
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
        X_collection = []
        Y_collection = []
        
        if resolution == '60m' and country == 'nl':
            if data_type == 'ind':
                houses = 30
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs=60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                print("Making pairs")
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 60 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 3,
                    num_houses = 22,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs=60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                
                for i in range(3):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 3 houses data, each with 60 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection
            
        if resolution == '60m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs=70,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                print("Making pairs")
               
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'overlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 70 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 1,
                    num_houses = 6,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs=60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                
                for i in range(1):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 1 houses data, each with 60 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection
                return X_collection, Y_collection

        if resolution == '30m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2*4,
                            num_pairs=70,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                print("Making pairs")
               
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'overlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 70 pairs, each pair has input length of {prediction_length*2*3} and output length of {prediction_length*2}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 1,
                    num_houses = 6,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2*4,
                            num_pairs=70,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                
                for i in range(1):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 1 houses data, each with 70 pairs, each pair has input length of {prediction_length*2*3} and output length of {prediction_length*2}')
                return X_collection, Y_collection
            
        if resolution == '15m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4*4,
                            num_pairs=40,
                            window_split_rato=0.75,
                            random_state=random_state)
                print("Making pairs")
               
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'overlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 40 pairs, each pair has input length of {prediction_length*4*3} and output length of {prediction_length*4}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 1,
                    num_houses = 6,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4*4,
                            num_pairs=40,
                            window_split_rato=0.75,
                            random_state=random_state)
                
                for i in range(1):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 1 houses data, each with 40 pairs, each pair has input length of {prediction_length*4*3} and output length of {prediction_length*4}')
                return X_collection, Y_collection
 
        if resolution == '30m' and country == 'uk':
            if data_type == 'ind':
                houses = 30
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2*4,
                            num_pairs= 60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                print("Making pairs")
               
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'overlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 60 pairs, each pair has input length of {prediction_length*2*3} and output length of {prediction_length*2}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 5,
                    num_houses = 100,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2*4,
                            num_pairs=60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                
                for i in range(5):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 5 houses data, each with 60 pairs, each pair has input length of {prediction_length*2*3} and output length of {prediction_length*2}')
                return X_collection, Y_collection
            
        if resolution == '60m' and country == 'uk':
            if data_type == 'ind':
                houses = 30
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs= 60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                print("Making pairs")
               
                for i in tqdm(range(houses)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    X, Y = pair_maker.make_pairs(df, 'overlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                
                print(f'Loaded {houses} houses data, each with 60 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection
            
            if data_type == 'agg':
                print("Loading aggregated data")
                df_train, _ = loader.load_dataset_agg(
                    num_agg = 5,
                    num_houses = 60,
                    random_state = random_state
                )
                print("Making pairs")
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*4,
                            num_pairs=60,
                            window_split_rato=window_split_rato,
                            random_state=random_state)
                
                for i in tqdm(range(5)):
                    df = df_train[df_train['id'] == df_train['id'].unique()[i]]
                    
                    X, Y = pair_maker.make_pairs(df, 'noverlap')
                    X = torch.tensor(X)
                    Y = torch.tensor(Y)
                    
                    X_collection.append(X)
                    Y_collection.append(Y)
                X_collection = torch.stack(X_collection)
                Y_collection = torch.stack(Y_collection)
                print(f'Loaded 5 houses data, each with 60 pairs, each pair has input length of {prediction_length*3} and output length of {prediction_length}')
                return X_collection, Y_collection

                        
if __name__ == "__main__":
    x, y = data_for_exp(
        resolution ='15m',
        country = 'ge',
        data_type = 'agg',
    )
    print(x.shape, y.shape)

                 