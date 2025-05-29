import time
import random
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.fml_utils import compute_loss_and_accuracy, split_dataset, DataSplitStrategy
from common.model.model_wrapper import ModelWrapper
from common.const import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, LEARNING_RATE
from common.enum.aggregation_method import AggregationMethod
from trainers.base_federated_trainer import BaseFederatedTrainer
from trainers.federated_sa.client_trainer import ClientTrainer, ClientConfig, SAConfig


class FederatedConfig:
    def __init__(self, sa_config: SAConfig, num_clients: int = NUM_CLIENTS, num_rounds: int = NUM_ROUNDS,
                 local_epochs: int = LOCAL_EPOCHS, learning_rate: float = LEARNING_RATE,
                 aggregation_method: AggregationMethod = AggregationMethod.FED_AVG,
                 fed_prox_mu: Optional[float] = None, data_split_strategy: DataSplitStrategy = DataSplitStrategy.STRATIFIED_EQUAL):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.sa_config = sa_config
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

        self.clients = [
            ClientTrainer(
                model_fn(),
                self.client_loaders[i],
                ClientConfig(
                    sa_config=config.sa_config,
                    learning_rate=config.learning_rate,
                    aggregation_method=config.aggregation_method,
                    fed_prox_mu=config.fed_prox_mu,
                ),
                config.num_clients,
                client_index=i
            )
            for i in range(config.num_clients)
        ]

        self.dropped_client_index = random.randint(0, config.num_clients - 1) if config.sa_config.drop_clients else None

        self.test_acc_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.train_loss_history = []

        self.adjust_rounds_for_fed_sgd(train_loader)

    def train(self) -> ModelWrapper:
        print(
            f"\n##### Running Federated Learning with Secure Aggregation | "
            f"num_clients={self.config.num_clients}, "
            f"num_rounds={self.config.num_rounds}, "
            f"local_epochs={self.config.local_epochs}, "
            f"learning_rate={self.config.learning_rate}, "
            f"aggregation={self.config.aggregation_method.value.upper()}, "
            f"mask_noise_scale={self.config.sa_config.mask_noise_scale}, "
            f"drop_clients={self.config.sa_config.drop_clients} #####\n"
        )

        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            print(f"======== Round {round_num + 1} ========")

            active_clients = self._select_active_clients()

            self._train_clients(active_clients)
            self._generate_masks(active_clients)
            self._simulate_mask_exchange(active_clients)
            self._apply_masks(active_clients)
            self._aggregate_masked_models(active_clients)
            self._evaluate_global_model(active_clients)

        exec_time = time.time() - start_time
        return ModelWrapper(self.global_model, self.train_acc_history, self.train_loss_history, self.test_acc_history, self.test_loss_history, exec_time)

    def _select_active_clients(self) -> List[ClientTrainer]:
        if self.dropped_client_index is not None:
            return [c for c in self.clients if c.client_index != self.dropped_client_index]
        return self.clients

    def _train_clients(self, clients: List[ClientTrainer]):
        for client in clients:
            client.model.load_state_dict(self.global_model.state_dict())
            if self.config.aggregation_method == AggregationMethod.FED_SGD:
                client.train_step_sgd()
                self.client_num_samples[client.client_index] = client.num_samples
            else:
                client.train()
            print(f"Client {client.client_index + 1} | Train Acc: {client.train_acc:.2f}% | Train Loss: {client.train_loss:.4f}")

    @staticmethod
    def _generate_masks(clients: List[ClientTrainer]):
        for client in clients:
            client.generate_masks()

    @staticmethod
    def _simulate_mask_exchange(clients: List[ClientTrainer]):
        for sender in clients:
            for receiver in clients:
                if sender.client_index != receiver.client_index:
                    receiver.receive_masks(sender.client_index, sender.masks[receiver.client_index])

    @staticmethod
    def _apply_masks(clients: List[ClientTrainer]):
        for client in clients:
            client.apply_masks()

    def _aggregate_masked_models(self, clients: List[ClientTrainer]):
        masked_updates = [client.get_masked_weights() for client in clients]
        method = self.config.aggregation_method

        if method in {AggregationMethod.FED_ADAM, AggregationMethod.FED_YOGI, AggregationMethod.FED_ADAGRAD}:
            self._aggregate_fed_opt(masked_updates, method)
        elif method == AggregationMethod.FED_SGD:
            self._aggregate_fed_sgd(masked_updates)
        else:
            self._aggregate_fed_avg(masked_updates)

    def _aggregate_fed_opt(self, updates: List[Dict[str, torch.Tensor]], method: AggregationMethod) -> None:
        self.initialize_adaptive_state_if_needed(self.global_model.state_dict())
        aggregated_weights = {
            k: torch.stack([update[k] for update in updates], dim=0).mean(dim=0)
            for k in updates[0]
        }

        global_weights = self.global_model.state_dict()
        for name in global_weights:
            delta = aggregated_weights[name] - global_weights[name]
            self.update_adaptive_parameter(name, delta, method, global_weights)

        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_avg(self, updates: List[Dict[str, torch.Tensor]]):
        aggregated_weights = {
            k: torch.stack([update[k] for update in updates], dim=0).mean(dim=0)
            for k in updates[0]
        }
        self.global_model.load_state_dict(aggregated_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_sgd(self, grads: List[Dict[str, torch.Tensor]]):
        global_weights = self.global_model.state_dict()
        for k in global_weights:
            grad_avg = torch.stack([g[k] for g in grads], dim=0).mean(dim=0)
            global_weights[k] -= self.config.learning_rate * grad_avg
        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _evaluate_global_model(self, clients: List[ClientTrainer]):
        acc, loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())
        self.test_acc_history.append(acc)
        self.test_loss_history.append(loss)
        for client in clients:
            self.train_acc_history.append(client.train_acc)
            self.train_loss_history.append(client.train_loss)
        print(f"Global Model | Test Acc: {acc:.2f}% | Test Loss: {loss:.4f}")
