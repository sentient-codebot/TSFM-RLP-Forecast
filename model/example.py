"""define an example linear model for time series forecasting
date: 2024-07-01
author: sentient-codebot
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.utils
from torch.utils.data import DataLoader
import numpy as np
from numpy import ndarray

import gluonts as ts
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
)

from tqdm import tqdm

from .typing import Float, Int, Callable

class GluonTSNetwork(ABC):
    @abstractmethod
    def hybrid_forward(self, F, past_target, future_target):
        raise NotImplementedError

class LinearModel(nn.Module):
    def __init__(self, past_length: int, prediction_length: int, hidden_dim: int = 50):
        super().__init__()
        self.past_length = past_length
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(past_length, hidden_dim),
            nn.Linear(hidden_dim, prediction_length)
        )
        
    def forward(self, x: Float[Tensor, 'batch, past_length']) -> Float[Tensor, 'batch, prediction_length']:
        return self.linear(x)
    
class ExampleTrainNetwork(GluonTSNetwork):
    def __init__(self, model: nn.Module, loss_function: Callable):
        self.model = model
        self.loss_fn = loss_function
        
    def hybrid_forward(self, past_target, future_target):
        self.model.train()
        prediction = self.model(past_target)
        loss = self.loss_fn(prediction, future_target) # make sure to return a scalar
        return loss
    
class ExamplePredNetwork(GluonTSNetwork):
    def __init__(self, model: nn.Module):
        self.model = model
        
    @torch.no_grad()
    def hybrid_forward(self, F, past_target, future_target):
        self.model.eval()
        prediction = self.model(past_target)
        return prediction
    
@dataclass
class TrainerOutput:
    dataloader: DataLoader
    network: ExampleTrainNetwork
    current_epoch: int
    epoch_loss: float
    
class ExampleTrainer():
    def __init__(self, lr: float, epochs: int, device: torch.device|None = None):
        self.lr = lr
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(
        self,
        network: ExampleTrainNetwork,
        dataloader: DataLoader,
    ):
        optimizer = torch.optim.Adam(network.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            epoch_loss = 0.
            for data_entry in dataloader: # data_entry is a dict
                optimizer.zero_grad()
                past_target = data_entry['past_target'].to(self.device)
                future_target = data_entry['future_target'].to(self.device)
                loss = network.hybrid_forward(past_target, future_target) # return loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            yield TrainerOutput(dataloader, network, epoch, epoch_loss)

"""the estimator class is central to the use of GluonTS models. 
it is responsible for:
- instantiating the training networks
- train the network and return a predictor (containing the prediction network)
"""
class ExampleEstimator():
    @validated()
    def __init__(
        self,
        prediction_length: int,
        past_length: int,
        hidden_dim: int,
        trainer: ExampleTrainer,
    ) -> None:
        self.prediction_length = prediction_length
        self.past_length = past_length
        self.hidden_dim = hidden_dim
        self.model = None
        self.train_network = None
        self.pred_network = None
        self.trainer = trainer
        
    def create_transformation(self):
        # skip for this example
        raise NotImplementedError
    
    def create_training_data_loader(self, dataset: ts.dataset.Dataset, batch_size: int=64, num_batches_per_epoch: int=50, **kwargs):
        mask_unobserved = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
        training_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1,
                min_future=self.prediction_length,
            ),
            past_length=self.past_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        dataloader = TrainDataLoader(
            Cached(dataset.train),
            batch_size=batch_size,
            stack_fn=batchify,
            transform=mask_unobserved + training_splitter,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        # next(iter(dataloader)).keys()
        # dict_keys(['start', 'feat_static_cat', 'item_id', 'past_observed_values', 'future_observed_values', 'past_target', 'future_target', 'past_is_pad', 'forecast_start'])
        # mainly use: 'past_target', 'future_target'
        return dataloader
    
    def create_training_network(self):
        self.model = LinearModel(self.past_length, self.prediction_length, self.hidden_dim)
        self.train_network = ExampleTrainNetwork(self.model, nn.MSELoss())
        return self.train_network
        
    def create_predictor(self, batch_size: int=32):
        assert self.model is not None, "model is not created yet"
        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        self.pred_network = ExamplePredNetwork(self.model)
        
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=['past_target'],
            prediction_net=self.pred_network,
            batch_size=batch_size,
            input_transform=prediction_splitter,
            forecast_generator=QuantileForecastGenerator(quantiles=[0.5]), # what should I use?
        )
        
    def train(self, dataset: ts.dataset.Dataset) -> None:
        assert self.train_network is not None, "training network is not created yet"
        dataloader = self.create_training_data_loader(dataset)
        it = self.trainer.train(
            self.train_network,
            dataloader,
        )
        for output in tqdm(it, total=self.trainer.epochs):
            # output is a TrainerOutput
            print(f"epoch {output.current_epoch}, loss: {output.epoch_loss}")
            
        print("training complete")