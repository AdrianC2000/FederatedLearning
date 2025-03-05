from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from common.model_wrapper import ModelWrapper
from common.simple_cnn import SimpleCNN
from common.utils import remove_module_prefix, compute_loss_and_accuracy
from trainers.trainer import Trainer
from trainers.federated_dp.client_trainer import ClientTrainer, ClientConfig, TrainingResult

class FederatedConfig:
    def __init__(self, num_clients: int, num_rounds: int = 15, local_epochs: int = 3, epsilon: Optional[float] = None):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.epsilon = epsilon

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

    def train(self) -> ModelWrapper:
        print(f"\n##### Running FL | num_clients={self.config.num_clients}, epsilon={self.config.epsilon}, num_rounds={self.config.num_rounds}, local_epochs={self.config.local_epochs} #####\n")

        for round_num in range(self.config.num_rounds):
            print(f"======== Round {round_num + 1} ========")
            client_weights_list = []

            for index in range(self.config.num_clients):
                self._train_single_client(client_weights_list, index)

            self._calculate_global_model_fed_avg(client_weights_list)
            global_test_acc, global_test_loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())

            self.test_acc_history.append(global_test_acc)
            self.test_loss_history.append(global_test_loss)

            print(f"Global Model | Test Acc: {global_test_acc:.2f}% | Test Loss: {global_test_loss:.4f}\n")

        return ModelWrapper(self.global_model, self.client_train_accuracies, self.client_train_losses, self.test_acc_history, self.test_loss_history)

    def _train_single_client(self, client_weights_list, index):
        client_model = SimpleCNN()
        # Load the global model weights into the client model
        client_model.load_state_dict(self.global_model.state_dict())
        client_model.train()

        client_config = ClientConfig(epochs=self.config.local_epochs, epsilon=self.config.epsilon)
        client_trainer = ClientTrainer(client_model, self.client_loaders[index], client_config)
        training_result: TrainingResult = client_trainer.train()

        client_weights_list.append(training_result.client_weights)

        self.client_train_accuracies.append(training_result.train_acc)
        self.client_train_losses.append(training_result.train_loss)

        print(
            f"Client {index + 1} | Train Acc: {training_result.train_acc:.2f}% | Train Loss: {training_result.train_loss:.4f}"
        )

    def _calculate_global_model_fed_avg(self, client_weights_list: List[dict]) -> None:
        global_dict = self.global_model.state_dict()
        client_weights_list = [remove_module_prefix(w) for w in client_weights_list]
        # Find common layers between the global model and client models
        shared_keys = set(global_dict.keys()).intersection(set(client_weights_list[0].keys()))

        for key in shared_keys:
            # Stack tensors from all clients from each layer and calculate the mean
            stacked_tensors = torch.stack([client_weights[key] for client_weights in client_weights_list])

            # Calculate the mean of the stacked tensors
            mean_tensor = stacked_tensors.mean(dim=0)
            if torch.sum(mean_tensor).item() != 0:
                global_dict[key] = mean_tensor

        # Update the global model with the averaged weights
        self.global_model.load_state_dict(global_dict, strict=False)
        self.global_model.eval()
