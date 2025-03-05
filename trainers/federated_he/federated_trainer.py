import time
from typing import Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import tenseal as ts
from common.model_wrapper import ModelWrapper
from common.simple_cnn import SimpleCNN
from common.utils import compute_loss_and_accuracy, remove_module_prefix
from trainers.federated_he.client_trainer import ClientConfig, ClientTrainer, TrainingResult, HEConfig
from trainers.trainer import Trainer

class FederatedConfig:
    def __init__(self, num_clients: int, num_rounds: int = 15, local_epochs: int = 3, he_config: Optional[HEConfig] = None):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.he_config = he_config

class FederatedTrainer(Trainer):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, config: FederatedConfig) -> None:
        self.config = config
        self.global_model = SimpleCNN()
        self.test_loader = test_loader

        split_sizes = [len(train_loader.dataset) // config.num_clients] * config.num_clients
        self.client_datasets = random_split(train_loader.dataset, split_sizes)
        self.client_loaders = [DataLoader(ds, batch_size=train_loader.batch_size, shuffle=False) for ds in self.client_datasets]

        self.client_train_accuracies: List[float] = []
        self.client_train_losses: List[float] = []
        self.test_acc_history: List[float] = []
        self.test_loss_history: List[float] = []

        self.context = self._setup_he_context(config.he_config) if config.he_config else None

    @staticmethod
    def _setup_he_context(he_config: HEConfig) -> ts.Context:
        context = ts.context(he_config.encryption_scheme, poly_modulus_degree=he_config.poly_modulus_degree, coeff_mod_bit_sizes=he_config.coeff_mod_bit_sizes)
        context.generate_galois_keys()
        return context

    def train(self) -> ModelWrapper:
        he_details = (
            f"HE=True, poly_modulus_degree={self.config.he_config.poly_modulus_degree}, "
            f"coeff_mod_bit_sizes={self.config.he_config.coeff_mod_bit_sizes}, "
            f"scale={self.config.he_config.scale}"
            if self.config.he_config else "HE=False"
        )

        print(f"\n##### Running FL | num_clients={self.config.num_clients}, num_rounds={self.config.num_rounds}, "
              f"local_epochs={self.config.local_epochs}, {he_details} #####\n")

        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            print(f"======== Round {round_num + 1} ========")
            client_weights_list = []

            for index in range(self.config.num_clients):
                self._train_single_client(client_weights_list, index)

            self._calculate_global_model_fed_avg(client_weights_list)
            global_test_acc, global_test_loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())

            self.test_acc_history.append(global_test_acc)
            self.test_loss_history.append(global_test_loss)
            print(f"Global Model | Test Acc: {global_test_acc:.2f}% | Test Loss: {global_test_loss:.2f}\n")

        exec_time = time.time() - start_time
        return ModelWrapper(self.global_model, self.client_train_accuracies, self.client_train_losses, self.test_acc_history, self.test_loss_history, exec_time)

    def _train_single_client(self, client_weights_list, index):
        client_model = SimpleCNN()
        client_model.load_state_dict(self.global_model.state_dict())
        client_model.train()

        client_config = ClientConfig(epochs=self.config.local_epochs, he_config=self.config.he_config)
        client_trainer = ClientTrainer(client_model, self.client_loaders[index], client_config, self.context)
        training_result: TrainingResult = client_trainer.train()

        if self.context:
            client_weights_list.append(training_result.encrypted_weights)
        else:
            client_weights_list.append(training_result.decrypted_weights)

        self.client_train_accuracies.append(training_result.train_acc)
        self.client_train_losses.append(training_result.train_loss)

        print(f"Client {index + 1} | Train Acc: {training_result.train_acc:.2f}% | Train Loss: {training_result.train_loss:.4f}")

    def _calculate_global_model_fed_avg(self, client_weights_list: List[dict]) -> None:
        if self.context:
            aggregated_weights = {}
            num_clients = len(client_weights_list)

            for k in client_weights_list[0]:
                encrypted_sum = client_weights_list[0][k]
                for i in range(1, num_clients):
                    encrypted_sum += client_weights_list[i][k]

                encrypted_avg = encrypted_sum * (1 / num_clients)
                aggregated_weights[k] = encrypted_avg

            decrypted_weights = {}
            for k in aggregated_weights:
                decrypted_values = aggregated_weights[k].decrypt()
                decrypted_tensor = torch.tensor(decrypted_values).reshape(self.global_model.state_dict()[k].shape)

                scale_factor = max(decrypted_tensor.abs().max().item(), 1e-15)
                decrypted_weights[k] = decrypted_tensor / scale_factor

            self.global_model.load_state_dict(decrypted_weights, strict=False)

        else:
            self.global_model.load_state_dict(remove_module_prefix(client_weights_list[0]), strict=False)

        self.global_model.eval()
