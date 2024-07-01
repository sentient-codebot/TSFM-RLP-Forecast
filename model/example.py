"""define an example linear model for time series forecasting
date: 2024-07-01
author: sentient-codebot
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from numpy import ndarray
import gluonts as ts
from gluonts.core.component import validated
import torch.utils
import torch.utils.data
from tqdm import tqdm

from .typing import Float, Int, Callable

class GluonTSNetwork(ABC):
    @abstractmethod
    def hybrid_forward(self, F, past_target, future_target):
        raise NotImplementedError

class LinearModel(nn.Module):
    def __init__(self, lookback_length: int, prediction_length: int, hidden_dim: int = 50):
        super().__init__()
        self.lookback_length = lookback_length
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(lookback_length, hidden_dim),
            nn.Linear(hidden_dim, prediction_length)
        )
        
    def forward(self, x: Float[Tensor, 'batch, lookback_length']) -> Float[Tensor, 'batch, prediction_length']:
        return self.linear(x)
    
class ExampleTrainNetwork(GluonTSNetwork):
    def __init__(self, model: nn.Module, loss_function: Callable):
        self.model = model
        self.loss_fn = loss_function
        
    def hybrid_forward(self, F, past_target, future_target):
        # NOTE: the F is merely following the official tutorial. 
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
    dataloader: torch.utils.data.DataLoader
    network: ExampleTrainNetwork
    current_epoch: int
    epoch_loss: float
    
class ExampleTrainer():
    def __init__(self, lr: float, epochs: int):
        self.lr = lr
        self.epochs = epochs
        
    def train(
        self,
        network: ExampleTrainNetwork,
        dataloader: torch.utils.data.DataLoader,
    ):
        optimizer = torch.optim.Adam(network.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            epoch_loss = 0.
            for past_target, future_target in dataloader:
                optimizer.zero_grad()
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
        lookback_length: int,
        hidden_dim: int,
        trainer: ExampleTrainer,
    ) -> None:
        self.prediction_length = prediction_length
        self.lookback_length = lookback_length
        self.hidden_dim = hidden_dim
        self.model = None
        self.train_network = None
        self.pred_network = None
        self.trainer = trainer
        
    def create_transformation(self):
        # skip for this example
        raise NotImplementedError
    
    def create_training_data_loader(self):
        # TODO
        # self.dataloader 
        raise NotImplementedError
    
    def create_training_network(self):
        self.model = LinearModel(self.lookback_length, self.prediction_length, self.hidden_dim)
        self.train_network = ExampleTrainNetwork(self.model, nn.MSELoss())
        return self.train_network
        
    def create_predictor(self, transformation):
        assert self.model is not None, "model is not created yet"
        ... # transformations
        self.pred_network = ExamplePredNetwork(self.model)
        
        # TODO: how exactly is a predictor defined?
        ... # return predictor
        
    def train(self) -> None:
        assert self.train_network is not None, "training network is not created yet"
        dataloader = self.create_training_data_loader()
        it = self.trainer.train(
            self.train_network,
            dataloader,
        )
        for output in tqdm(it, total=self.trainer.epochs):
            # output is a TrainerOutput
            print(f"epoch {output.current_epoch}, loss: {output.epoch_loss}")
            
        print("training complete")