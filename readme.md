# Time Series Foundation Models for Residential Load Profile Forecasting

This project aims to compare the zero-shot forecast performance of Time Series Foundation Models (TSFMs) on various scales of residential load profile (RLP) forecasting problems. 

This is the official repository of paper: [Link to paper](). 

## Logging Result Procedure

Example: *exp/chronos/chronos_predictor.py*

1. import `utility.configuration` (as `cf`) and `exp.eva_metrics` (as `evm`) modules.
2. generate a unique `exp_id` for each run. `exp_id=cf.generate_time_id()`
3. for each sub-run (e.g. with different country, resolution.)
   1. log data configuration `data_config=cf.DataConfig(country='nl',...)`
   2. log model configuration `model_config=cf.ModelConfig(model='chronos',...)`
   3. log evaluation results `eval_metrics=evm.EvaluationMetrics(...)`
   4. integrate into `exp_config=cf.ExperimentConfig(exp_id=exp_id, data=data_config, model=model_config, eval_metrics=eval_metrics)`
   5. save to .csv `exp_config.append_csv(f'result/{exp_id}.csv')`

## Structure

### File Structure

- dataset: contains class definition of datasets used in the project. 
  - XXX.py: class definition of dataset XXX.
  - ...: data preprocessing. 
- model: contains the (wrapper) class of the TSFMs used in the project.
  - YYY.py: class definition of TSFM YYY.
- utility: contains utility functions used in the project, including data 
  - argument_parser.py: argument parser for the project.
  - configuration.py: configuration class definition. 
- configs: .yaml configuration files. 

### Configuration Usage

The configuration module defines a basic configuration class that can be extended to contain configuration settings for data, model, etc. The base class allows for easy conversion between dictionary, configuration object, and .yaml file. 

#### Configuration Hierachy

- ExperimentConfig
  - general experiment-specific settings such as `exp_id`.
  - data: DataConfig. configuration for data.
  - model: ModelConfig. configuration for model.
  - ...