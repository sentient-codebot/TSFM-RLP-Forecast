"""
...
"""
import yaml

import dataset.data_process as dp


def main():
    reso_country = [
            ('30m', 'uk'),
            ('60m', 'uk'),
    ]
    for reso, country in reso_country:
        _type = 'ind'
        print('--------------------------------------------------')
        print(f"reso: {reso}, country: {country}, type: {_type}")
        print('--------------------------------------------------')
        # load datastet
        loader = dp.LoadDataset(
            resolution=reso,
            country=country,
            split_ratio=1
        )
        dict_row_group_ids = loader.get_row_group_ids()

        with open(f'{reso}_{country}_split_ids.yaml', 'w') as file:
            yaml.dump(dict_row_group_ids, file, default_flow_style=False)
            
        # count num_uniques
        foorbar = []
        for key in dict_row_group_ids.keys():
            foorbar += dict_row_group_ids[key]
        num_uniques = len(set(foorbar))
        print(f"num_uniques: {num_uniques}")
    
if __name__ == "__main__":
    main()