import math
import time
import torch
import torch.nn as nn
from typing import List, Dict
from torch.utils.data import DataLoader

import tenseal as ts

from common.const import NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS, FED_PROX_MU, LEARNING_RATE
from common.enum.aggregation_method import AggregationMethod
from common.fml_utils import compute_loss_and_accuracy, split_dataset_stratified
from common.model.model_wrapper import ModelWrapper
from trainers.base_federated_trainer import BaseFederatedTrainer
from trainers.federated_he.client_trainer import ClientTrainer, ClientConfig, HEConfig


class FederatedConfig:
    def __init__(
            self,
            he_config: HEConfig,
            num_clients: int = NUM_CLIENTS,
            num_rounds: int = NUM_ROUNDS,
            local_epochs: int = LOCAL_EPOCHS,
            learning_rate: float = LEARNING_RATE,
            aggregation_method: AggregationMethod = AggregationMethod.FED_AVG,
            fed_prox_mu: float = FED_PROX_MU,
            seed: int = 42,
    ):
        self.he_config = he_config
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method
        self.fed_prox_mu = fed_prox_mu
        self.seed = seed


class FederatedTrainer(BaseFederatedTrainer):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, model_fn: callable, config: FederatedConfig):
        super().__init__(config)
        self.config = config
        self.model_fn = model_fn
        self.global_model = model_fn()
        self.test_loader = test_loader

        self.he_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.config.he_config.poly_modulus_degree,
            coeff_mod_bit_sizes=self.config.he_config.coeff_mod_bit_sizes,
        )
        self.he_context.global_scale = self.config.he_config.scale
        self.he_context.generate_galois_keys()

        datasets = split_dataset_stratified(train_loader.dataset, config.num_clients, config.seed)
        self.client_loaders = [DataLoader(ds, batch_size=train_loader.batch_size, shuffle=False) for ds in datasets]

        self.client_train_accuracies = []
        self.client_train_losses = []
        self.test_acc_history = []
        self.test_loss_history = []

        self.adaptive_optimizer_state = {
            "momentum_buffers": {},
            "variance_buffers": {},
            "step_count": 0
        }

        self.adjust_rounds_for_fed_sgd(train_loader)

    def train(self) -> ModelWrapper:
        he = self.config.he_config
        print(
            f"\n##### Running FL with HE | num_clients={self.config.num_clients}, "
            f"num_rounds={self.config.num_rounds}, local_epochs={self.config.local_epochs}, "
            f"learning_rate={self.config.learning_rate}, aggregation={self.config.aggregation_method.value.upper()}, "
            f"poly_modulus_degree={he.poly_modulus_degree}, coeff_mod_bit_sizes={he.coeff_mod_bit_sizes}, "
            f"scale=2^{int(round(math.log2(he.scale)))}"
            f"{f', fed_prox_mu={self.config.fed_prox_mu}' if self.config.aggregation_method == AggregationMethod.FED_PROX else ''} #####\n"
        )
        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            print(f"======== Round {round_num + 1} ========")
            client_updates = [self._train_single_client(i) for i in range(self.config.num_clients)]

            self._aggregate(client_updates)
            self._evaluate_global_model()

        exec_time = time.time() - start_time

        return ModelWrapper(
            self.global_model,
            self.client_train_accuracies,
            self.client_train_losses,
            self.test_acc_history,
            self.test_loss_history,
            exec_time
        )

    def _train_single_client(self, index: int) -> Dict[str, ts.CKKSVector]:
        client_model = self.model_fn()
        client_model.load_state_dict(self.global_model.state_dict())

        client_config = ClientConfig(
            context=self.he_context,
            epochs=self.config.local_epochs,
            learning_rate=self.config.learning_rate,
            aggregation_method=self.config.aggregation_method,
            fed_prox_mu=self.config.fed_prox_mu if self.config.aggregation_method == AggregationMethod.FED_PROX else None
        )

        trainer = ClientTrainer(client_model, self.client_loaders[index], client_config)
        result = trainer.train_step_sgd() if self.config.aggregation_method == AggregationMethod.FED_SGD else trainer.train()

        self.client_train_accuracies.append(result.train_acc)
        self.client_train_losses.append(result.train_loss)

        print(f"Client {index + 1} | Train Acc: {result.train_acc:.2f}% | Train Loss: {result.train_loss:.4f}")
        return result.encrypted_client_updates

    def _aggregate(self, client_updates: List[Dict[str, ts.CKKSVector]]) -> None:
        method = self.config.aggregation_method
        if method in {AggregationMethod.FED_ADAM, AggregationMethod.FED_YOGI, AggregationMethod.FED_ADAGRAD}:
            self._aggregate_adaptive_from_avg(client_updates, method)
        elif method == AggregationMethod.FED_SGD:
            self._aggregate_fed_sgd(client_updates)
        else:
            self._aggregate_fed_avg(client_updates)

    def _aggregate_adaptive_from_avg(self, encrypted_weights_list: List[Dict[str, ts.CKKSVector]], method: AggregationMethod) -> None:
        aggregated_weights = self._decrypt_and_average(encrypted_weights_list)
        self.initialize_adaptive_state_if_needed(self.global_model.state_dict())
        self.adaptive_optimizer_state["step_count"] += 1
        step = self.adaptive_optimizer_state["step_count"]

        global_weights = self.global_model.state_dict()
        for name in global_weights:
            delta = aggregated_weights[name] - global_weights[name]

            self.update_adaptive_parameter(
                name=name,
                delta=delta,
                method=method,
                global_weights=global_weights,
                step=step
            )

        self.global_model.load_state_dict(global_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_avg(self, encrypted_weights_list: List[Dict[str, ts.CKKSVector]]) -> None:
        aggregated_weights = self._decrypt_and_average(encrypted_weights_list)
        self.global_model.load_state_dict(aggregated_weights, strict=False)
        self.global_model.eval()

    def _aggregate_fed_sgd(self, encrypted_gradients_list: List[Dict[str, ts.CKKSVector]]) -> None:
        averaged_gradients = self._decrypt_and_average(encrypted_gradients_list)
        global_state = self.global_model.state_dict()
        for key, grad_tensor in averaged_gradients.items():
            global_state[key].sub_(self.config.learning_rate * grad_tensor)
        self.global_model.eval()

    def _decrypt_and_average(self, encrypted_list: List[Dict[str, ts.CKKSVector]]) -> Dict[str, torch.Tensor]:
        num_clients = len(encrypted_list)
        result = {}

        for key in encrypted_list[0]:
            agg = encrypted_list[0][key].copy()
            for client_dict in encrypted_list[1:]:
                agg += client_dict[key]
            agg = agg * (1.0 / num_clients)
            decrypted = agg.decrypt()
            result[key] = torch.tensor(decrypted, dtype=torch.float32).view_as(self.global_model.state_dict()[key])

        return result

    def _evaluate_global_model(self) -> None:
        acc, loss = compute_loss_and_accuracy(self.global_model, self.test_loader, nn.CrossEntropyLoss())
        self.test_acc_history.append(acc)
        self.test_loss_history.append(loss)
        print(f"Global Model | Test Acc: {acc:.2f}% | Test Loss: {loss:.4f}")
