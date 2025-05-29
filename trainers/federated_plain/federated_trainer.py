import time
import torch
import torch.nn as nn
from typing import List, Dict
from torch.utils.data import DataLoader

from common.const import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, FED_PROX_MU, LEARNING_RATE
from common.enum.aggregation_method import AggregationMethod
from common.fml_utils import compute_loss_and_accuracy, remove_module_prefix, split_dataset, DataSplitStrategy
from common.model.model_wrapper import ModelWrapper
from trainers.base_federated_trainer import BaseFederatedTrainer
from trainers.federated_plain.client_trainer import ClientTrainer, ClientConfig


class FederatedConfig:
    def __init__(self, num_clients: int = NUM_CLIENTS, num_rounds: int = NUM_ROUNDS, local_epochs: int = LOCAL_EPOCHS,
                 learning_rate: float = LEARNING_RATE, aggregation_method: AggregationMethod = AggregationMethod.FED_AVG,
                 fed_prox_mu: float = FED_PROX_MU, data_split_strategy: DataSplitStrategy = DataSplitStrategy):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method
        self.fed_prox_mu = fed_prox_mu
        self.data_split_strategy = data_split_strategy

class FederatedTrainer(BaseFederatedTrainer):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, model_fn: callable, config: FederatedConfig):
        super().__init__(config)
        self.config = config
        self.model_fn = model_fn
        self.global_model = model_fn()
        self.test_loader = test_loader

        datasets = split_dataset(train_loader.dataset, config.num_clients, config.data_split_strategy)
        self.client_loaders = [DataLoader(ds, batch_size=train_loader.batch_size, shuffle=False) for ds in datasets]
        self.client_num_samples = [len(ds) for ds in datasets]

        self.client_train_accuracies = []
        self.client_train_losses = []
        self.test_acc_history = []
        self.test_loss_history = []

        self.adjust_rounds_for_fed_sgd(train_loader)

    def train(self) -> ModelWrapper:
        print(f"\n##### Running Federated Learning | aggregation={self.config.aggregation_method.value.upper()} #####")
        start_time = time.time()

        for round_index in range(self.config.num_rounds):
            print(f"======== Round {round_index + 1} ========")
            client_updates = [self._train_single_client(client_id) for client_id in range(self.config.num_clients)]
            self._aggregate(client_updates)
            self._evaluate_global_model()

        exec_time = time.time() - start_time

        return ModelWrapper(self.global_model, self.client_train_accuracies, self.client_train_losses,
                            self.test_acc_history, self.test_loss_history, exec_time)

    def _train_single_client(self, client_id: int) -> dict:
        client_model = self.model_fn()
        client_model.load_state_dict(self.global_model.state_dict())

        client_config = ClientConfig(
            epochs=self.config.local_epochs,
            learning_rate=self.config.learning_rate,
            aggregation_method=self.config.aggregation_method,
            fed_prox_mu=self.config.fed_prox_mu if self.config.aggregation_method == AggregationMethod.FED_PROX else None
        )
        trainer = ClientTrainer(client_model, self.client_loaders[client_id], client_config)

        if self.config.aggregation_method == AggregationMethod.FED_SGD:
            result = trainer.train_step_sgd()
            self.client_num_samples[client_id] = result.num_samples
        else:
            result = trainer.train()

        self.client_train_accuracies.append(result.train_acc)
        self.client_train_losses.append(result.train_loss)

        print(f"Client {client_id + 1} | Train Acc: {result.train_acc:.2f}% | Train Loss: {result.train_loss:.4f}")
        return result.client_update

    def _aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        method = self.config.aggregation_method
        if method == AggregationMethod.FED_SGD:
            self._aggregate_fed_sgd(client_updates)
        elif method in {AggregationMethod.FED_ADAM, AggregationMethod.FED_YOGI, AggregationMethod.FED_ADAGRAD}:
            self._aggregate_fed_opt(client_updates, method)
        else:
            self._aggregate_fed_avg(client_updates)

    def _aggregate_fed_avg(self, client_weights: List[Dict[str, torch.Tensor]]) -> None:
        global_weights = self.global_model.state_dict()
        client_weights = [remove_module_prefix(weights) for weights in client_weights]

        total_samples = sum(self.client_num_samples)
        for parameter_name in global_weights:
            weighted = [
                weights[parameter_name] * (n / total_samples)
                for weights, n in zip(client_weights, self.client_num_samples)
            ]
            global_weights[parameter_name] = sum(weighted)

        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_sgd(self, client_gradients: List[Dict[str, torch.Tensor]]) -> None:
        global_weights = self.global_model.state_dict()
        client_gradients = [remove_module_prefix(grad) for grad in client_gradients]

        total_samples = sum(self.client_num_samples)
        for parameter_name in global_weights:
            weighted = [
                gradient[parameter_name] * (n / total_samples)
                for gradient, n in zip(client_gradients, self.client_num_samples)
            ]
            averaged_gradient = sum(weighted)
            global_weights[parameter_name] -= self.config.learning_rate * averaged_gradient

        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_opt(self, client_updates: List[Dict[str, torch.Tensor]], method: AggregationMethod) -> None:
        self.initialize_adaptive_state_if_needed(self.global_model.state_dict())

        global_weights = self.global_model.state_dict()
        total_samples = sum(self.client_num_samples)

        for name in global_weights:
            delta = sum([
                (update[name] - global_weights[name]) * (n / total_samples)
                for update, n in zip(client_updates, self.client_num_samples)
            ], start=torch.zeros_like(global_weights[name]))

            self.update_adaptive_parameter(name, delta, method, global_weights)

        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _evaluate_global_model(self) -> None:
        accuracy, loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())
        self.test_acc_history.append(accuracy)
        self.test_loss_history.append(loss)
        print(f"Global Model | Test Acc: {accuracy:.2f}% | Test Loss: {loss:.4f}")
