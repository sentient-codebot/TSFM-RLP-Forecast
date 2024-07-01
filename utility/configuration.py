from dataclasses import dataclass, fields
from typing import Tuple
import os
from abc import abstractmethod
from functools import partial
from argparse import ArgumentParser, Namespace

import yaml

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
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)
        
@dataclass
class DataConfig(BaseConfig):
    dataset: str
    root: str
    # resolution: str
    # load: bool
    # normalize: bool
    
@dataclass
class ExampleDataConfig(DataConfig):
    param_only_for_this_dataset: str
        
@dataclass
class ModelConfig(BaseConfig):
    model_class: str # e.g., lag-llama, chronos
    # some_common_model_settings: str
    
@dataclass
class ExampleModelConfig(ModelConfig):
    param_only_for_this_model: int = 6
    
@dataclass
class ExperimentConfig(BaseConfig):
    exp_id: str
    data: DataConfig
    model: ModelConfig
    # log_wandb: bool = False
    
    subconfigs = {
        'data': DataConfig,
        'model': ModelConfig,
    }
    
    @classmethod
    def init_subconfig(cls, subconfig_name, subconfig_dict) -> BaseConfig|dict:
        if subconfig_name not in cls.subconfigs:
            return subconfig_dict
        if subconfig_name == 'data':
            if subconfig_dict['dataset'] == 'example':
                return ExampleDataConfig.from_dict(subconfig_dict)
            else:
                return DataConfig.from_dict(subconfig_dict)
        if subconfig_name == 'model':
            if subconfig_dict['model_class'] == 'example_model':
                return ExampleModelConfig.from_dict(subconfig_dict)
            else:
                return ModelConfig.from_dict(subconfig_dict)
    
if __name__ == "__main__":
    model_config = ModelConfig.from_yaml("model_config.yaml")
    data_config = DataConfig(dataset="example", root="root")
    
    exp_config = ExperimentConfig(exp_id=1,
                                    model=model_config,
                                    data=data_config)
    print(exp_config.model)
    exp_config.to_yaml("exp_config.yaml")
    pass