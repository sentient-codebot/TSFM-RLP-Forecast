from dataclasses import dataclass, fields
from typing import Tuple
import os
import sys
from abc import abstractmethod
from functools import partial
from argparse import ArgumentParser, Namespace
from typing import Union, Any
import yaml
from datetime import datetime
import random

import pandas as pd

def flatten_dict(dict_obj, parent_key='', sep=''):
    out_dict = {}
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            value = flatten_dict(value, parent_key + sep + key, sep='_')
            out_dict.update(value)
        else:
            out_dict[parent_key + sep + key] = value
    return out_dict

@dataclass
class BaseConfig:
    subconfigs = {}

    @classmethod
    @abstractmethod
    def init_subconfig(cls, subconfig_name, subconfig_dict):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, kwargs: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k not in valid_keys:
                continue
            if k in cls.subconfigs.keys():
                if isinstance(v, dict):
                    filtered_kwargs[k] = cls.init_subconfig(k, v)
                elif isinstance(v, cls.subconfigs[k]):
                    filtered_kwargs[k] = v
                else:
                    raise ValueError(f"Invalid type for {k}")
            else:
                filtered_kwargs[k] = v

        return cls(**filtered_kwargs)

    def to_dict(self):
        out_dict = {}
        for key, item in self.__dict__.items():
            if key in self.subconfigs and isinstance(item, BaseConfig):
                out_dict[key] = item.to_dict()
            else:
                out_dict[key] = item
        return out_dict

    @classmethod
    def inherit(cls, parent, **kwargs):
        parent_dict = parent.to_dict()
        parent_dict.update(kwargs)
        return cls.from_dict(parent_dict)

    @classmethod
    def from_yaml(cls, path:str):
        with open(path, 'r') as f:
            _obj = cls.from_dict(yaml.safe_load(f))
            _obj.load_path = path

        return _obj

    def to_yaml(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)

    def append_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            pd.DataFrame([flatten_dict(self.to_dict())]).to_csv(path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([flatten_dict(self.to_dict())]).to_csv(path, index=False)

    def to_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame([flatten_dict(self.to_dict())]).to_csv(path, index=False)

    def to_stdout(self):
        pd.DataFrame([flatten_dict(self.to_dict())]).to_csv(sys.stdout, index=False)

@dataclass
class DataConfig(BaseConfig):
    country: str
    aggregation_type: str
    resolution: str
    note: str = ""

@dataclass
class ModelConfig(BaseConfig):
    model_name: str
    lookback_window: int
    prediction_length: int

@dataclass
class ExperimentConfig(BaseConfig):
    exp_id: str
    data: DataConfig
    model: ModelConfig
    result: Any = ''
    # log_wandb: bool = False

    subconfigs = {
        'data': DataConfig,
        'model': ModelConfig,
        'result': Any,
    }

    @classmethod
    def init_subconfig(cls, subconfig_name, subconfig_dict) -> Union[BaseConfig, dict]:
        if subconfig_name not in cls.subconfigs:
            return subconfig_dict
        if subconfig_name == 'data':
            return DataConfig.from_dict(subconfig_dict)
        if subconfig_name == 'model':
            return ModelConfig.from_dict(subconfig_dict)

def generate_time_id():
    return datetime.now().strftime("%Y%m%d" + "-" + f"{random.randint(0, 9999):04d}")

if __name__ == "__main__":
    model_config = ModelConfig(model_name="chronos-t5-tiny", lookback_window=72, prediction_length=24)
    data_config = DataConfig(country="nl", aggregation_type="ind", resolution="60m")

    exp_config = ExperimentConfig(exp_id=generate_time_id(),
                                    model=model_config,
                                    data=data_config)
    print(exp_config.model)
    exp_config.to_yaml("exp_config.yaml")
    pass