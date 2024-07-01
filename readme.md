# Time Series Foundation Models for Residential Load Profile Forecasting

This project aims to compare the zero-shot forecast performance of Time Series Foundation Models (TSFMs) on various scales of residential load profile (RLP) forecasting problems. 

This is the official repository of paper: [Link to paper](). 

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