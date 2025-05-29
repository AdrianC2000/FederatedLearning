import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict

from common.fml_utils import compute_loss_and_accuracy, remove_module_prefix
from common.const import LEARNING_RATE, MOMENTUM, LOCAL_EPOCHS, SHARE_NOISE_SCALE
from common.enum.aggregation_method import AggregationMethod
from trainers.base_trainer import BaseTrainer


class SMPCConfig:
    def __init__(self, share_noise_scale: float = SHARE_NOISE_SCALE, drop_clients: bool = False):
        self.share_noise_scale = share_noise_scale
        self.drop_clients = drop_clients


class ClientConfig:
    def __init__(self, smpc_config: SMPCConfig, learning_rate: float = LEARNING_RATE, momentum: float = MOMENTUM,
                 epochs: int = LOCAL_EPOCHS, aggregation_method: AggregationMethod = AggregationMethod.FED_AVG,
                 fed_prox_mu: Optional[float] = None):
        self.smpc_config = smpc_config
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.aggregation_method = aggregation_method
        self.fed_prox_mu = fed_prox_mu


class ClientTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: ClientConfig,
                 num_clients: int, client_index: int):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.num_clients = num_clients
        self.client_index = client_index

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)
        self.criterion = nn.CrossEntropyLoss()

        self.shares: Dict[int, Dict[str, torch.Tensor]] = {}
        self.received_shares: Dict[int, Dict[str, torch.Tensor]] = {}

        self.train_loss: float = 0.0
        self.train_acc: float = 0.0
        self.num_samples: int = 0

        self.weights_before: Optional[Dict[str, torch.Tensor]] = None
        self.update: Optional[Dict[str, torch.Tensor]] = None
        self.masked_update: Optional[Dict[str, torch.Tensor]] = None

    def train(self):
        self.model.train()
        self.weights_before = {name: param.detach().clone() for name, param in self.model.named_parameters()}
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

    def _compute_prox_term(self) -> torch.Tensor:
        prox_term = 0.0
        for name, param in self.model.named_parameters():
            if name in self.weights_before:
                prox_term += ((param - self.weights_before[name]) ** 2).sum()
        return prox_term

    def train_step_sgd(self):
        self.model.train()
        self.weights_before = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        x, y = next(iter(self.train_loader))
        self.num_samples = x.size(0)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        self.train_acc, self.train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)
        weights_after = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        self.update = {
            name: (self.weights_before[name] - weights_after[name]) / self.config.learning_rate
            for name in weights_after
        }

    def generate_shares(self):
        for j in range(self.num_clients):
            self.shares[j] = {
                k: self.update[k] / self.num_clients + torch.randn_like(v) * self.config.smpc_config.share_noise_scale
                for k, v in self.update.items()
            }

    def receive_share(self, sender_index: int, share: Dict[str, torch.Tensor]):
        self.received_shares[sender_index] = share

    def compute_masked_update(self):
        combined = {k: torch.zeros_like(v) for k, v in self.update.items()}

        for j in self.received_shares:
            for k in combined:
                combined[k] += self.received_shares[j][k]

        self_share = self.shares[self.client_index]
        for k in combined:
            combined[k] += self_share[k]

        self.masked_update = combined

    def get_masked_weights(self) -> Dict[str, torch.Tensor]:
        return self.masked_update
