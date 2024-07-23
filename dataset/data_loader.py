import torch
import dataset.data_process as dp
from tqdm import tqdm


def data_for_exp(
        resolution: str = '60m',
        country: str = 'nl',
        data_type: str = 'ind',
        prediction_length: int = 64,
        window_split_rato: float = 0.5,
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
            # ('30m','aus'),
            # ('60m','aus')
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
                houses = 70
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
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
                
                print('Loaded 70 houses data, each with 60 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
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
                print('Loaded 3 houses data, each with 60 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection
            
        if resolution == '60m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
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
                
                print(f'Loaded {houses} houses data, each with 70 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
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
                print('Loaded 1 houses data, each with 60 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection

        if resolution == '30m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
                            num_pairs=140,
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
                
                print(f'Loaded {houses} houses data, each with 140 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
                            num_pairs=120,
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
                print('Loaded 1 houses data, each with 120 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection
            
        if resolution == '15m' and country == 'ge':
            if data_type == 'ind':
                houses = 6
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
                            num_pairs=70*4,
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
                
                print(f'Loaded {houses} houses data, each with 280 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
                            num_pairs=60*4,
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
                print('Loaded 1 houses data, each with 240 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection
 
        if resolution == '30m' and country == 'uk':
            if data_type == 'ind':
                houses = 100
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
                            num_pairs= 100,
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
                
                print(f'Loaded {houses} houses data, each with 100 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
                            num_pairs=100,
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
                print('Loaded 5 houses data, each with 100 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection
            
        if resolution == '60m' and country == 'uk':
            if data_type == 'ind':
                houses = 100
                print("Loading individual data")
                df_train, _ = loader.load_dataset_ind()
                pair_maker = dp.PairMaker(
                            window_length=prediction_length*2,
                            num_pairs= 100,
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
                
                print(f'Loaded {houses} houses data, each with 100 pairs, each pair has input length of 64 and output length of 64')
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
                            window_length=prediction_length*2,
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
                print('Loaded 5 houses data, each with 60 pairs, each pair has input length of 64 and output length of 64')
                return X_collection, Y_collection

                        
if __name__ == "__main__":
    x, y = data_for_exp(
        resolution ='60m',
        country = 'nl',
        data_type = 'ind',
    )
    print(x.shape, y.shape)

                 