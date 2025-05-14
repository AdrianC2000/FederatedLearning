import random
import time

import numpy as np
import torch

from common.result_utils.fl_result_utils import FLResultUtils
from trainers.federated_plain.federated_trainer import FederatedTrainer, FederatedConfig
from trainers.federated_plain.client_trainer import AggregationMethod
from common.model.model_wrapper import ModelWrapper

def run_plain_experiments(
        train_loader,
        test_loader,
        model_fn,
        dataset: str,
        num_clients: int,
        num_rounds: int,
        local_epochs: int,
        num_runs: int,
        learning_rate: float,
        base_seed: int
) -> dict[str, ModelWrapper]:
    print(f"\n[Plain] Running {num_runs} run(s) for each aggregation method")
    global_start = time.time()

    methods = [AggregationMethod.FED_AVG, AggregationMethod.FED_SGD, AggregationMethod.FED_ADAGRAD, AggregationMethod.FED_ADAM, AggregationMethod.FED_YOGI]
    fed_prox_mus = [0.001, 0.01, 0.1]
    results = {}

    def avg_metric(runs: list[list[float]]) -> list[float]:
        return [sum(xs) / len(xs) for xs in zip(*runs)]

    for method in methods:
        aggregation = method.value.lower()
        print(f"→ {aggregation}")
        local_start = time.time()

        test_acc_runs, test_loss_runs = [], []
        exec_times = []

        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            print(f"[Plain] Run {run_idx + 1}/{num_runs} | seed = {run_seed}")

            torch.manual_seed(run_seed)
            random.seed(run_seed)
            np.random.seed(run_seed)

            trainer = FederatedTrainer(
                train_loader,
                test_loader,
                model_fn,
                FederatedConfig(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    aggregation_method=method,
                    learning_rate=learning_rate
                )
            )
            model = trainer.train()
            test_acc_runs.append(model.test_acc)
            test_loss_runs.append(model.test_loss)
            exec_times.append(model.exec_time)

        avg_model = ModelWrapper(
            model=model.model,
            train_acc=[0.0],
            train_loss=[0.0],
            test_acc=avg_metric(test_acc_runs),
            test_loss=avg_metric(test_loss_runs),
            exec_time=sum(exec_times) / num_runs
        )

        FLResultUtils.save(avg_model, dataset=dataset, method="federated_plain", aggregation=aggregation)
        FLResultUtils.save_metadata(
            config={"type": f"plain_{aggregation}"},
            dataset=dataset,
            method="federated_plain",
            aggregation=aggregation
        )
        results[aggregation] = avg_model

        local_elapsed = time.time() - local_start
        print(f"{aggregation.upper()} training finished in {int(local_elapsed // 60):02}:{int(local_elapsed % 60):02} for {num_runs} run(s)")

    for mu in fed_prox_mus:
        aggregation = "fed_prox"
        mu_label = f"mu_{str(mu).replace('.', '_')}"
        print(f"→ FED_PROX (mu={mu})")
        local_start = time.time()

        test_acc_runs, test_loss_runs = [], []
        exec_times = []

        for _ in range(num_runs):
            trainer = FederatedTrainer(
                train_loader,
                test_loader,
                model_fn,
                FederatedConfig(
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    aggregation_method=AggregationMethod.FED_PROX,
                    fed_prox_mu=mu
                )
            )
            model = trainer.train()
            test_acc_runs.append(model.test_acc)
            test_loss_runs.append(model.test_loss)
            exec_times.append(model.exec_time)

        avg_model = ModelWrapper(
            model=model.model,
            train_acc=[0.0],
            train_loss=[0.0],
            test_acc=avg_metric(test_acc_runs),
            test_loss=avg_metric(test_loss_runs),
            exec_time=sum(exec_times) / num_runs
        )

        FLResultUtils.save(avg_model, dataset=dataset, method="federated_plain", aggregation=aggregation, mu=mu)
        FLResultUtils.save_metadata(
            config={"type": "plain_fedprox", "fed_prox_mu": mu},
            dataset=dataset,
            method="federated_plain",
            aggregation=aggregation,
            mu=mu
        )
        results[mu_label] = avg_model

        local_elapsed = time.time() - local_start
        print(f"FED_PROX (μ={mu}) training finished in {int(local_elapsed // 60):02}:{int(local_elapsed % 60):02} for {num_runs} run(s)")

    total_elapsed = time.time() - global_start
    print(f"All Plain experiments finished in {int(total_elapsed // 60):02}:{int(total_elapsed % 60):02} for {num_runs} run(s)")
    return results
