import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from trainers.base_trainer import BaseTrainer
from common.const import LEARNING_RATE, MOMENTUM, LOCAL_EPOCHS
from common.fml_utils import compute_loss_and_accuracy, remove_module_prefix
from common.enum.aggregation_method import AggregationMethod


class ClientConfig:
    def __init__(self, learning_rate: float = LEARNING_RATE, momentum: float = MOMENTUM, epochs: int = LOCAL_EPOCHS,
                 aggregation_method: AggregationMethod = AggregationMethod.FED_AVG, fed_prox_mu: Optional[float] = None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.aggregation_method = aggregation_method
        self.fed_prox_mu = fed_prox_mu


class TrainingResult:
    def __init__(self, client_update: dict, train_loss: float, train_acc: float, num_samples: Optional[int] = None):
        self.client_update = client_update  # weights or gradients
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.num_samples = num_samples


class ClientTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: ClientConfig) -> None:
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum
        )
        self.global_weights = (
            {name: param.detach().clone() for name, param in self.model.named_parameters()}
            if self.config.aggregation_method == AggregationMethod.FED_PROX else None
        )

    def train(self) -> TrainingResult:
        self.model.train()

        for _ in range(self.config.epochs):
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                if self.config.aggregation_method == AggregationMethod.FED_PROX:
                    loss += (self.config.fed_prox_mu / 2) * self._compute_prox_term()

                loss.backward()
                self.optimizer.step()

        train_acc, train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)
        weights_after = remove_module_prefix(self.model.state_dict())

        return TrainingResult(weights_after, train_loss, train_acc)

    def _compute_prox_term(self) -> float:
        prox_term = 0.0
        for name, param in self.model.named_parameters():
            if name in self.global_weights:
                global_param = self.global_weights[name]
                prox_term += ((param - global_param) ** 2).sum()
        return prox_term

    def train_step_sgd(self) -> TrainingResult:
        self.model.train()

        weights_before = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        x, y = next(iter(self.train_loader))
        num_samples = x.size(0)

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        train_acc, train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)
        weights_after = remove_module_prefix(self.model.state_dict())

        gradients = {
            name: (weights_before[name] - weights_after[name]) / self.config.learning_rate
            for name in weights_before
        }

        return TrainingResult(gradients, train_loss, train_acc, num_samples)
