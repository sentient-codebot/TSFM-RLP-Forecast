"""define an example linear model for time series forecasting
date: 2024-07-01
author: sentient-codebot
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.utils
from torch.utils.data import DataLoader
import numpy as np
from numpy import ndarray
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Callable, Optional
import gluonts as ts
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import QuantileForecastGenerator, DistributionForecastGenerator
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
)

from tqdm import tqdm

from .typing import Float, Int, Callable

class ForecastType(enum.Enum):
    POINT = enum.auto()
    QUANTILE = enum.auto()
    STUDENTT = enum.auto()

class GluonTSNetwork(ABC, nn.Module):
    # @abstractmethod
    # def forward(self, *args, **kwargs):
    #     raise NotImplementedError
    pass

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 50, make_output_head: Optional[Callable] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        if make_output_head is None:
            self.output_head = nn.Identity()
        else:
            self.output_head = make_output_head()
        
    def forward(self, x: Float[Tensor, 'batch, input_dim']) -> Float[Tensor, 'batch, output_dim']:
        return self.output_head(self.linear(x))
    
class ExampleTrainNetwork(GluonTSNetwork):
    def __init__(self, model: nn.Module, loss_function: Callable, forecast_type: ForecastType = ForecastType.POINT):
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.forecast_type = forecast_type
        if forecast_type == ForecastType.POINT:
            # model assert output_dim == prediction_length
            pass
        elif forecast_type == ForecastType.STUDENTT:
            # model assert output_dim == prediction_length * 3
            self.distr_output = StudentTOutput()
        else:
            raise NotImplementedError(f"forecast {self.forecast_type} not implemented")
        
    def forward(self, past_target, future_target):
        self.model.train()
        prediction = self.model(past_target)
        if self.forecast_type == ForecastType.POINT:
            loss = self.loss_fn(prediction, future_target) # make sure to return a scalar
        elif self.forecast_type == ForecastType.STUDENTT:
            # prediction: tuple of (df, loc, scale)
            loss = -self.distr_output.distribution(*prediction).log_prob(future_target).mean()
            # NOTE: student_t distr_args (df, loc, scale) have loc and scale, BUT distribution's argument is (distr_args, loc, scale), which also has loc and scale.
        return loss
    
    # TODO: WHY tf does GluonTS make "hybrid_forward" AN MANDATORY METHOD????? 
    # NOTE: current version: removed the hybrid_forward method. it is now the forward method. works totally fine. wtf wrote that tutorial? 
    # def forward(self, *args, **kwargs):
    #     return self.hybrid_forward(*args, **kwargs)
    
class ExamplePredNetwork(GluonTSNetwork):
    def __init__(self, model: nn.Module, forecast_type: ForecastType = ForecastType.POINT):
        super().__init__()
        self.model = model
        self.forecast_type = forecast_type
        
    @torch.no_grad()
    def forward(self, past_target):
        self.model.eval()
        prediction = self.model(past_target)
        if self.forecast_type == ForecastType.POINT:
            return rearrange(prediction, 'b l -> b () l')
        elif self.forecast_type == ForecastType.STUDENTT:
            return prediction
        else:
            raise NotImplementedError(f"forecast {self.forecast_type} not implemented")
    
    # def forward(self, *args, **kwargs):
    #     return self.hybrid_forward(*args, **kwargs)
    
def make_student_t_output_head(input_dim: int, prediction_length: int) -> nn.Module:
    # input shape: [batch, input_dim]
    class _pack_output(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return (x,) # pack into tuple, to accom for the ugly forecast generator, which calls distr_output.distribution(*prediction)
    
    return nn.Sequential(
        Rearrange('b (l q) -> b l q', l=prediction_length), # assert l * q == input_dim
        StudentTOutput().get_args_proj(in_features=input_dim//prediction_length),
        _pack_output(),
    )
    
@dataclass
class TrainerOutput:
    dataloader: DataLoader
    network: ExampleTrainNetwork
    current_epoch: int
    epoch_loss: float
    
class ExampleTrainer():
    def __init__(self, lr: float, epochs: int, device: Optional[torch.device] = None):
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
            epoch_count = 0
            for data_entry in dataloader: # data_entry is a dict
                optimizer.zero_grad()
                past_target = data_entry['past_target'].to(self.device)
                future_target = data_entry['future_target'].to(self.device)
                loss = network.forward(past_target, future_target) # return loss
                loss.backward()
                nn.utils.clip_grad_norm_(network.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_count += past_target.shape[0]
            epoch_loss /= epoch_count # average per sample
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
        forecast_type: ForecastType,
        trainer: ExampleTrainer,
    ) -> None:
        self.prediction_length = prediction_length
        self.past_length = past_length
        self.hidden_dim = hidden_dim
        self.model = None
        self.train_network = None
        self.pred_network = None
        self.trainer = trainer
        self.forecast_type = forecast_type
        
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
    
    def create_model(self):
        if self.forecast_type == ForecastType.POINT:
            self.model = MLPModel(self.past_length, self.prediction_length, self.hidden_dim)
        elif self.forecast_type == ForecastType.STUDENTT:
            self.model = MLPModel(self.past_length, self.prediction_length * self.hidden_dim, self.hidden_dim, 
                                     make_output_head=lambda: make_student_t_output_head(self.prediction_length * self.hidden_dim, self.prediction_length))
        return self.model
    
    def create_training_network(self):
        self.model = self.create_model()
        self.train_network = ExampleTrainNetwork(self.model, nn.MSELoss(), self.forecast_type)
        return self.train_network
        
    def create_predictor(self, batch_size: int=32):
        assert self.model is not None, "model is not created yet"
        mask_unobserved = AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
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
        self.pred_network = ExamplePredNetwork(self.model, self.forecast_type)
        
        if self.forecast_type == ForecastType.POINT:
            return PyTorchPredictor(
                prediction_length=self.prediction_length,
                input_names=['past_target'],
                prediction_net=self.pred_network,
                batch_size=batch_size,
                input_transform=mask_unobserved + prediction_splitter,
                forecast_generator=QuantileForecastGenerator(quantiles=[0.5]),
                # output_transform=lambda input, model_pred: rearrange(model_pred, 'b l -> b () l'), # [batch, prediction_length] -> [batch, 1, prediction_length]
            ) # output_transform: repeat unsqueeze for each quantile
        elif self.forecast_type == ForecastType.STUDENTT:
            return PyTorchPredictor(
                prediction_length=self.prediction_length,
                input_names=['past_target'],
                prediction_net=self.pred_network,
                batch_size=batch_size,
                input_transform=mask_unobserved + prediction_splitter,
                forecast_generator=DistributionForecastGenerator(StudentTOutput()),
            ) 
        
    def train(self, dataset: ts.dataset.Dataset) -> None:
        assert self.train_network is not None, "training network is not created yet"
        dataloader = self.create_training_data_loader(dataset)
        it = self.trainer.train(
            self.train_network,
            dataloader,
        )
        pbar = tqdm(it, total=self.trainer.epochs)
        for output in pbar:
            # output is a TrainerOutput
            pbar.set_description(f"epoch {output.current_epoch}, loss: {output.epoch_loss:.4f}")
            
        print("training complete")
        
        return self.create_predictor()