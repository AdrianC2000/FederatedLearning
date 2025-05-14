import time
from typing import List
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from common.model.model_wrapper import ModelWrapper
from common.fml_utils import compute_loss_and_accuracy
from common.const import LEARNING_RATE, MOMENTUM, CENTRALISED_EPOCHS
from trainers.base_trainer import BaseTrainer


class CentralizedConfig:
    def __init__(self,
                 learning_rate: float = LEARNING_RATE,
                 momentum: float = MOMENTUM,
                 epochs: int = CENTRALISED_EPOCHS):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs

class CentralizedTrainer(BaseTrainer):
    def __init__(self, model_fn: callable, train_loader: DataLoader, test_loader: DataLoader, config: CentralizedConfig) -> None:
        self.model = model_fn()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        self.train_acc_history: List[float] = []
        self.train_loss_history: List[float] = []
        self.test_acc_history: List[float] = []
        self.test_loss_history: List[float] = []

    def train(self) -> ModelWrapper:
        start_time = time.time()

        for epoch in range(self.config.epochs):
            train_acc, train_loss = self._train_one_epoch()
            test_acc, test_loss = compute_loss_and_accuracy(self.model, self.test_loader, self.criterion)

            self.train_acc_history.append(train_acc)
            self.train_loss_history.append(train_loss)
            self.test_acc_history.append(test_acc)
            self.test_loss_history.append(test_loss)

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}"
            )

        exec_time = time.time() - start_time

        return ModelWrapper(
            model=self.model,
            train_acc=self.train_acc_history,
            train_loss=self.train_loss_history,
            test_acc=self.test_acc_history,
            test_loss=self.test_loss_history,
            exec_time=exec_time
        )

    def _train_one_epoch(self) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return 100.0 * correct / total, running_loss / len(self.train_loader)
