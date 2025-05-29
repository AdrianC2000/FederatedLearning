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
from trainers.federated_dp.client_trainer import ClientTrainer, ClientConfig, DPConfig


class FederatedConfig:
    def __init__(self, dp_config: DPConfig, num_clients: int = NUM_CLIENTS, num_rounds: int = NUM_ROUNDS,
                 local_epochs: int = LOCAL_EPOCHS, learning_rate: float = LEARNING_RATE,
                 aggregation_method: AggregationMethod = AggregationMethod.FED_AVG, fed_prox_mu: float = FED_PROX_MU,
                 data_split_strategy: DataSplitStrategy = DataSplitStrategy.STRATIFIED_EQUAL):
        self.dp_config = dp_config
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
        dp = self.config.dp_config
        print(f"\n##### Running FL with DP | clients={self.config.num_clients}, rounds={self.config.num_rounds}, "
              f"epochs={self.config.local_epochs}, agg={self.config.aggregation_method.value.upper()}, "
              f"ε={dp.epsilon}, δ={dp.delta}, max_grad_norm={dp.max_grad_norm}, lr={self.config.learning_rate}, "
              f"{f'fed_prox_mu={self.config.fed_prox_mu}' if self.config.aggregation_method == AggregationMethod.FED_PROX else ''} #####\n")

        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            print(f"======== Round {round_num + 1} ========")
            client_updates = [self._train_single_client(i) for i in range(self.config.num_clients)]
            self._aggregate(client_updates)
            self._evaluate_global_model()

        exec_time = time.time() - start_time
        return ModelWrapper(self.global_model, self.client_train_accuracies, self.client_train_losses,
                            self.test_acc_history, self.test_loss_history, exec_time)

    def _train_single_client(self, index: int) -> dict:
        client_model = self.model_fn()
        client_model.load_state_dict(self.global_model.state_dict())

        client_config = ClientConfig(
            dp_config=self.config.dp_config, epochs=self.config.local_epochs, learning_rate=self.config.learning_rate,
            aggregation_method=self.config.aggregation_method,
            fed_prox_mu=self.config.fed_prox_mu if self.config.aggregation_method == AggregationMethod.FED_PROX else None
        )
        trainer = ClientTrainer(client_model, self.client_loaders[index], client_config)

        if self.config.aggregation_method == AggregationMethod.FED_SGD:
            result = trainer.train_step_sgd()
            self.client_num_samples[index] = result.num_samples
        else:
            result = trainer.train()

        self.client_train_accuracies.append(result.train_acc)
        self.client_train_losses.append(result.train_loss)
        print(f"Client {index + 1} | Train Acc: {result.train_acc:.2f}% | Train Loss: {result.train_loss:.4f}")
        return result.client_update

    def _aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        method = self.config.aggregation_method
        if method == AggregationMethod.FED_SGD:
            self._aggregate_fed_sgd(client_updates)
        elif method in {AggregationMethod.FED_ADAM, AggregationMethod.FED_YOGI, AggregationMethod.FED_ADAGRAD}:
            self._aggregate_fed_opt(client_updates, method)
        else:
            self._aggregate_fed_avg(client_updates)

    def _aggregate_fed_avg(self, weights_list: List[Dict[str, torch.Tensor]]) -> None:
        global_dict = self.global_model.state_dict()
        weights = [remove_module_prefix(w) for w in weights_list]
        total_samples = sum(self.client_num_samples)
        for key in global_dict:
            weighted = [w[key] * (n / total_samples) for w, n in zip(weights, self.client_num_samples)]
            global_dict[key] = sum(weighted)
        self.global_model.load_state_dict(global_dict, strict=False)
        self.global_model.eval()

    def _aggregate_fed_sgd(self, gradients_list: List[Dict[str, torch.Tensor]]) -> None:
        global_dict = self.global_model.state_dict()
        grads = [remove_module_prefix(g) for g in gradients_list]
        total_samples = sum(self.client_num_samples)
        for key in global_dict:
            weighted = [g[key] * (n / total_samples) for g, n in zip(grads, self.client_num_samples)]
            grad_avg = sum(weighted)
            global_dict[key] -= self.config.learning_rate * grad_avg
        self.global_model.load_state_dict(global_dict, strict=False)
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
        acc, loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())
        self.test_acc_history.append(acc)
        self.test_loss_history.append(loss)
        print(f"Global Model | Test Acc: {acc:.2f}% | Test Loss: {loss:.4f}")
