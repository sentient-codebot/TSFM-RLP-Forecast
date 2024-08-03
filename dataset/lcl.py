"""Prepare GluonTS format dataset for LCL London
author: sentient-codebot
date: 2024-07-03
"""
import os

import pandas as pd
from gluonts.dataset.split import split
from gluonts.dataset.pandas import PandasDataset

def create_lcl():
    dir = "data/heatodiff-data/lcl_electricity_dataset/raw"
    csv_files = []
    for file in os.listdir(dir):
        # match the patter "LCL-*.csv"
        if file.startswith("LCL-") and file.endswith(".csv"):
            # process the file
            file_path = os.path.join(dir, file)
            csv_files.append(file_path)
    # load one csv
    _csv = csv_files[0]
    df = pd.read_csv(_csv, index_col=2, parse_dates=True)
    print(df.head())
    
    # sanity check: are the intervals consistent?
    # ddf = df[df['LCLid'] == 'MAC005555']
    # diffs = ddf.index.to_series().diff().dropna()
    # print(diffs.value_counts())
    
    # format "stdorToU" column to real
    df['isToU'] = df['stdorToU'].map({'Std': 0, 'ToU': 1})
    df.drop(columns=['stdorToU'], inplace=True)
    # deal with missing values
    "we group by LCLid and reindex each of the grouped dataframes."
    max_end = max(df.groupby('LCLid').apply(lambda _df: _df.index[-1]))
    dfs_dict = {}
    for item_id, gdf in df.groupby('LCLid'):
        # remove duplicate
        gdf = gdf[~gdf.index.duplicated(keep='first')]
        # reindex
        new_index = pd.date_range(gdf.index[0], end=max_end, freq='30T')
        dfs_dict[item_id] = gdf.reindex(new_index).drop('LCLid', axis=1)

    ds = PandasDataset(dfs_dict, target='KWH/hh (per half hour) ', feat_dynamic_real=['isToU']) # each key is an item id
    
    # train/test split
    prediction_length = 3 * 48
    ds_train, test_template = split(
        ds,
        offset = -1440
    )
    test_pairs = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=3,
    )
    return ds

def main():
    # scan legit csv files
    dir = "data/heatodiff-data/lcl_electricity_dataset/raw"
    csv_files = []
    for file in os.listdir(dir):
        # match the patter "LCL-*.csv"
        if file.startswith("LCL-") and file.endswith(".csv"):
            # process the file
            file_path = os.path.join(dir, file)
            csv_files.append(file_path)
    # load one csv
    _csv = csv_files[0]
    df = pd.read_csv(_csv, index_col=2, parse_dates=True)
    print(df.head())
    
    # sanity check: are the intervals consistent?
    # ddf = df[df['LCLid'] == 'MAC005555']
    # diffs = ddf.index.to_series().diff().dropna()
    # print(diffs.value_counts())
    
    # format "stdorToU" column to real
    df['isToU'] = df['stdorToU'].map({'Std': 0, 'ToU': 1})
    df.drop(columns=['stdorToU'], inplace=True)
    # deal with missing values
    "we group by LCLid and reindex each of the grouped dataframes."
    max_end = max(df.groupby('LCLid').apply(lambda _df: _df.index[-1]))
    dfs_dict = {}
    for item_id, gdf in df.groupby('LCLid'):
        # remove duplicate
        gdf = gdf[~gdf.index.duplicated(keep='first')]
        # reindex
        new_index = pd.date_range(gdf.index[0], end=max_end, freq='30T')
        dfs_dict[item_id] = gdf.reindex(new_index).drop('LCLid', axis=1)

    ds = PandasDataset(dfs_dict, target='KWH/hh (per half hour) ', feat_dynamic_real=['isToU']) # each key is an item id
    
    # train/test split
    prediction_length = 3 * 48
    ds_train, test_template = split(
        ds,
        offset = -1440
    )
    test_pairs = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=3,
    )
    pass
    
if __name__ == "__main__":
    main()