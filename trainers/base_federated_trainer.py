import math
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.utils.data import DataLoader

from common.enum.aggregation_method import AggregationMethod
from common.model.model_wrapper import ModelWrapper


class BaseFederatedTrainer(ABC):

    __BETA_1 = 0.9
    __BETA_2 = 0.99
    __EPSILON = 1e-8

    def __init__(self, config):
        self.config = config
        self.adaptive_optimizer_state = {
            "momentum_buffers": {},
            "variance_buffers": {}
        }

    @abstractmethod
    def train(self) -> ModelWrapper:
        pass

    def adjust_rounds_for_fed_sgd(self, train_loader: DataLoader):
        if self.config.aggregation_method != AggregationMethod.FED_SGD:
            return

        reference_epochs = self.config.local_epochs
        reference_rounds = self.config.num_rounds
        batch_size = train_loader.batch_size
        client_dataset_size = len(self.client_loaders[0].dataset)
        num_batches = math.ceil(client_dataset_size / batch_size)

        total_reference_steps = reference_epochs * num_batches * reference_rounds

        print(
            f"[FedSGD] Adjusting num_rounds from {self.config.num_rounds} to {total_reference_steps} "
            f"to match approx. {reference_rounds} rounds of {reference_epochs} epochs on {num_batches} batches."
        )

        self.config.num_rounds = total_reference_steps

    def initialize_adaptive_state_if_needed(self, global_weights: Dict[str, torch.Tensor]) -> None:
        if self.adaptive_optimizer_state["momentum_buffers"]:
            return
        for name in global_weights:
            self.adaptive_optimizer_state["momentum_buffers"][name] = torch.zeros_like(global_weights[name])
            self.adaptive_optimizer_state["variance_buffers"][name] = torch.zeros_like(global_weights[name])


    def update_adaptive_parameter(self, name: str, delta: torch.Tensor, method: AggregationMethod,
                                  global_weights: Dict[str, torch.Tensor]) -> None:
        momentum = self.adaptive_optimizer_state["momentum_buffers"]
        variance = self.adaptive_optimizer_state["variance_buffers"]

        new_momentum = self.__BETA_1 * momentum[name] + (1 - self.__BETA_1) * delta
        momentum[name] = new_momentum

        new_variance = self._update_variance(variance[name], delta, method)
        variance[name] = new_variance

        update_step = new_momentum / (new_variance.sqrt() + self.__EPSILON)
        global_weights[name] += self.config.learning_rate * update_step


    def _update_variance(self, previous: torch.Tensor, delta: torch.Tensor, method: AggregationMethod) -> torch.Tensor:
        if method == AggregationMethod.FED_ADAM:
            return self.__BETA_2 * previous + (1 - self.__BETA_2) * delta.pow(2)
        elif method == AggregationMethod.FED_YOGI:
            return previous - (1 - self.__BETA_2) * torch.sign(previous - delta.pow(2)) * delta.pow(2)
        elif method == AggregationMethod.FED_ADAGRAD:
            return previous + delta.pow(2)
        else:
            raise ValueError(f"Unsupported adaptive aggregation method: {method}")