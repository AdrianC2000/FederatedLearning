import math
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.utils.data import DataLoader

from common.enum.aggregation_method import AggregationMethod
from common.model.model_wrapper import ModelWrapper


class BaseFederatedTrainer(ABC):

    def __init__(self, config):
        self.config = config
        self.adaptive_optimizer_state = {
            "momentum_buffers": {},
            "variance_buffers": {},
            "step_count": 0
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


    def update_adaptive_parameter(self,
                                  name: str,
                                  delta: torch.Tensor,
                                  method: AggregationMethod,
                                  global_weights: Dict[str, torch.Tensor],
                                  step: int):
        beta1 = 0.9
        beta2 = 0.99
        epsilon = 1e-8
        lr = self.config.learning_rate

        momentum = self.adaptive_optimizer_state["momentum_buffers"]
        variance = self.adaptive_optimizer_state["variance_buffers"]

        new_momentum = beta1 * momentum[name] + (1 - beta1) * delta
        momentum[name] = new_momentum

        new_variance = self._update_variance(variance[name], delta, method, beta2)
        variance[name] = new_variance

        m_hat = new_momentum / (1 - beta1 ** step)
        v_hat = new_variance / (1 - beta2 ** step) if method != AggregationMethod.FED_ADAGRAD else new_variance

        update_step = m_hat / (v_hat.sqrt() + epsilon)
        global_weights[name] += lr * update_step


    @staticmethod
    def _update_variance(previous: torch.Tensor, delta: torch.Tensor,
                         method: AggregationMethod, beta2: float) -> torch.Tensor:
        if method == AggregationMethod.FED_ADAM:
            return beta2 * previous + (1 - beta2) * delta.pow(2)
        elif method == AggregationMethod.FED_YOGI:
            return previous - (1 - beta2) * torch.sign(previous - delta.pow(2)) * delta.pow(2)
        elif method == AggregationMethod.FED_ADAGRAD:
            return previous + delta.pow(2)
        else:
            raise ValueError(f"Unsupported adaptive aggregation method: {method}")