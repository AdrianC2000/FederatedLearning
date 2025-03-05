from typing import Optional, List
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tenseal as ts
from common.utils import remove_module_prefix, compute_loss_and_accuracy
from trainers.trainer import Trainer
from tenseal import SCHEME_TYPE

class HEConfig:
    def __init__(self, poly_modulus_degree: int = 32768,
                 coeff_mod_bit_sizes: list = [60, 40, 40, 60],
                 scale: float = 2**35):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.scale = scale
        self.encryption_scheme = SCHEME_TYPE.CKKS

class ClientConfig:
    def __init__(self, learning_rate: float = 0.003, momentum: float = 0.9, epochs: int = 3, he_config: Optional[HEConfig] = None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.he_config = he_config

class TrainingResult:
    def __init__(self, encrypted_weights: Optional[dict], decrypted_weights: Optional[dict], train_loss: float, train_acc: float):
        self.encrypted_weights = encrypted_weights  # For HE
        self.decrypted_weights = decrypted_weights  # For turned off HE
        self.train_loss = train_loss
        self.train_acc = train_acc

class ClientTrainer(Trainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: ClientConfig, context: Optional[ts.Context] = None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        self.context = self._setup_he_context(config.he_config) if config.he_config else None
        if context:
            self.context = context

    @staticmethod
    def _setup_he_context(he_config: HEConfig) -> ts.Context:
        context = ts.context(
            scheme=SCHEME_TYPE.CKKS,
            poly_modulus_degree=he_config.poly_modulus_degree,
            coeff_mod_bit_sizes=he_config.coeff_mod_bit_sizes
        )

        context.generate_galois_keys()
        return context

    def _encrypt_model_weights(self) -> dict:
        encrypted_weights = {}

        for name, param in self.model.state_dict().items():
            values = param.view(-1).tolist()
            encrypted_weights[name] = ts.ckks_vector(self.context, values, scale=self.config.he_config.scale)

        return encrypted_weights

    def train(self) -> TrainingResult:
        self.model.train()
        client_weights_before_he = remove_module_prefix(self.model.state_dict())

        for _ in range(self.config.epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        train_acc, train_loss = compute_loss_and_accuracy(self.model, self.train_loader, self.criterion)

        encrypted_weights = self._encrypt_model_weights() if self.context else None
        decrypted_weights = client_weights_before_he if not self.context else None
        return TrainingResult(encrypted_weights, decrypted_weights, train_loss, train_acc)