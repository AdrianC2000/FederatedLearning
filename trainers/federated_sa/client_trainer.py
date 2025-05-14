import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict

from common.fml_utils import compute_loss_and_accuracy, remove_module_prefix
from common.const import LEARNING_RATE, MOMENTUM, LOCAL_EPOCHS, MASK_NOISE_SCALE
from common.enum.aggregation_method import AggregationMethod
from trainers.base_trainer import BaseTrainer


class SAConfig:
    def __init__(self, mask_noise_scale: float = MASK_NOISE_SCALE, drop_clients: bool = False):
        self.mask_noise_scale = mask_noise_scale
        self.drop_clients = drop_clients


class ClientConfig:
    def __init__(
            self,
            sa_config: SAConfig,
            learning_rate: float = LEARNING_RATE,
            momentum: float = MOMENTUM,
            epochs: int = LOCAL_EPOCHS,
            aggregation_method: AggregationMethod = AggregationMethod.FED_AVG,
            fed_prox_mu: Optional[float] = None,
    ):
        self.sa_config = sa_config
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.aggregation_method = aggregation_method
        self.fed_prox_mu = fed_prox_mu


class ClientTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: ClientConfig, num_clients: int, client_index: int):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.num_clients = num_clients
        self.client_index = client_index

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        self.criterion = nn.CrossEntropyLoss()

        self.masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self.received_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self.train_loss: float = 0.0
        self.train_acc: float = 0.0

        self.weights_before: Optional[Dict[str, torch.Tensor]] = None
        self.update: Optional[Dict[str, torch.Tensor]] = None
        self.masked_update: Optional[Dict[str, torch.Tensor]] = None

    def train(self):
        self.model.train()
        self.weights_before = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        for _ in range(self.config.epochs):
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                if self.config.aggregation_method == AggregationMethod.FED_PROX:
                    loss += (self.config.fed_prox_mu / 2) * self._compute_prox_term()

                loss.backward()
                self.optimizer.step()

        self.train_acc, self.train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)
        weights_after = remove_module_prefix(self.model.state_dict())
        self.update = weights_after

    def train_step_sgd(self):
        self.model.train()
        self.weights_before = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        x, y = next(iter(self.train_loader))
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        self.train_acc, self.train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)

        weights_after = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        self.update = {
            name: (self.weights_before[name] - weights_after[name]) / self.config.learning_rate
            for name in weights_after
        }

    def _compute_prox_term(self) -> float:
        prox_term = 0.0
        for name, param in self.model.named_parameters():
            if name in self.weights_before:
                prox_term += ((param - self.weights_before[name]) ** 2).sum()
        return prox_term

    def generate_masks(self):
        weights = self.model.state_dict()
        for j in range(self.num_clients):
            if j == self.client_index:
                continue
            self.masks[j] = {
                k: torch.randn_like(v) * self.config.sa_config.mask_noise_scale
                for k, v in weights.items()
            }

    def receive_masks(self, sender_index: int, mask: Dict[str, torch.Tensor]):
        self.received_masks[sender_index] = mask

    def apply_masks(self):
        masked_update = {}
        for k in self.update:
            m_plus = sum((self.masks[j][k] for j in self.masks if k in self.masks[j]), torch.zeros_like(self.update[k]))
            m_minus = sum((self.received_masks[j][k] for j in self.received_masks if k in self.received_masks[j]), torch.zeros_like(self.update[k]))
            masked_update[k] = self.update[k] + m_plus - m_minus
        self.masked_update = masked_update

    def get_masked_weights(self) -> Dict[str, torch.Tensor]:
        return self.masked_update
