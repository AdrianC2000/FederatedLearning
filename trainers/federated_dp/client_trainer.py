from typing import Optional
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from trainers.trainer import Trainer
from common.utils import remove_module_prefix, compute_loss_and_accuracy

class ClientConfig:
    def __init__(self, learning_rate: float = 0.003, momentum: float = 0.9, epsilon: Optional[float] = None,
                 epochs: int = 3, max_grad_norm: float = 1.0, delta: float = 1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.delta = delta

class TrainingResult:
    def __init__(self, model: nn.Module, client_weights: dict, train_loss: float, train_acc: float):
        self.model = model
        self.client_weights = client_weights
        self.train_loss = train_loss
        self.train_acc = train_acc

class ClientTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: ClientConfig) -> None:
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        if self.config.epsilon is not None:
            self._enable_privacy()

    def _enable_privacy(self):
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=self.config.epsilon,
            target_delta=self.config.delta, # TODO -> research about this param
            max_grad_norm=self.config.max_grad_norm, # TODO -> research about this param
            epochs=self.config.epochs,
        )

    def train(self) -> TrainingResult:
        self.model.train()
        client_weights_before_dp = remove_module_prefix(self.model.state_dict())

        for _ in range(self.config.epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        train_acc, train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)
        return TrainingResult(self.model, client_weights_before_dp, train_loss, train_acc)
