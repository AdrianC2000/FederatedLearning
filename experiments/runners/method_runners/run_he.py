import os
import random
import numpy as np
import pandas as pd
from typing import Optional

import torch
from torch.utils.data import DataLoader

from trainers.federated_he.client_trainer import HEConfig
from trainers.federated_he.federated_trainer import FederatedTrainer, FederatedConfig
from common.result_utils.fl_result_utils import FLResultUtils
from common.result_utils.visualisation_utils import VisualisationUtils
from common.enum.aggregation_method import AggregationMethod
from common.enum.metric_type import MetricType
from common.model.model_wrapper import ModelWrapper


def run_he_experiments(
        aggregation_method: AggregationMethod,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_fn: callable,
        dataset: str,
        num_runs: int,
        num_clients: int,
        num_rounds: int,
        local_epochs: int,
        learning_rate: float,
        baseline_results: dict[str, ModelWrapper],
        fed_prox_mu: Optional[float] = None,
        seed: int = 42
):
    method = "homomorphic_encryption"
    privacy_method = "Homomorphic Encryption"
    aggregation = aggregation_method.value.lower()
    powers = [45]

    if aggregation_method == AggregationMethod.FED_PROX and fed_prox_mu is not None:
        mu_key = f"mu_{str(fed_prox_mu).replace('.', '_')}"
        relevant_baselines = {
            label: model for label, model in baseline_results.items()
            if label == "centralized" or label == mu_key
        }
    elif aggregation_method == AggregationMethod.FED_AVG:
        relevant_baselines = {
            label: model for label, model in baseline_results.items()
            if label in {"centralized", "fed_avg"}
        }
    elif aggregation_method == AggregationMethod.FED_SGD:
        relevant_baselines = {
            label: model for label, model in baseline_results.items()
            if label in {"centralized", "fed_sgd"}
        }
    else:
        relevant_baselines = {
            label: model for label, model in baseline_results.items()
            if label == "centralized"
        }

    result_models = relevant_baselines.copy()
    result_models_by_mu = {
        "mu_0_001": relevant_baselines.copy(),
        "mu_0_01": relevant_baselines.copy(),
        "mu_0_1": relevant_baselines.copy(),
    } if aggregation_method == AggregationMethod.FED_PROX else {}

    summary_rows_by_config = {}

    def avg_metric(runs: list[list[float]]) -> list[float]:
        return [sum(xs) / len(xs) for xs in zip(*runs)]

    for i, power in enumerate(powers, start=1):
        config = HEConfig(scale=2**power)
        config_label = f"scale_2^{power}"
        print(f"[{i}/{len(powers)}] Running HE config: {config_label}")

        test_acc_runs, test_loss_runs = [], []
        exec_times = []

        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            print(f"[HE] Run {run_idx + 1}/{num_runs} | seed = {run_seed}")

            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            random.seed(run_seed)

            trainer = FederatedTrainer(
                train_loader,
                test_loader,
                model_fn,
                FederatedConfig(
                    he_config=config,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    local_epochs=local_epochs,
                    learning_rate=learning_rate,
                    aggregation_method=aggregation_method,
                    fed_prox_mu=fed_prox_mu,
                    seed=run_seed
                ),
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
            exec_time=sum(exec_times) / num_runs,
        )

        result_models[config_label] = avg_model

        if aggregation_method == AggregationMethod.FED_PROX and fed_prox_mu is not None:
            mu_key = f"mu_{str(fed_prox_mu).replace('.', '_')}"
            if mu_key in result_models_by_mu:
                result_models_by_mu[mu_key][f"{mu_key}__{config_label}"] = avg_model

        FLResultUtils.save(avg_model, dataset, method, aggregation, config_label, fed_prox_mu)
        FLResultUtils.save_metadata(
            config={"scale": config.scale, **({"fed_prox_mu": fed_prox_mu} if fed_prox_mu else {})},
            dataset=dataset,
            method=method,
            aggregation=aggregation,
            config_name=config_label,
            mu=fed_prox_mu,
        )

        summary_rows = []
        for r, (acc, loss) in enumerate(zip(avg_model.test_acc, avg_model.test_loss), start=1):
            summary_rows.append({
                "scale": f"2^{power}",
                "round": r,
                "test_acc": acc,
                "test_loss": loss,
                "exec_time": avg_model.exec_time,
            })

        summary_rows_by_config[config_label] = summary_rows

    plot_dir = os.path.join("results", dataset, method, aggregation,
                            f"mu_{str(fed_prox_mu).replace('.', '_')}" if fed_prox_mu else "", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    suffix = VisualisationUtils.get_aggregation_title(aggregation_method)
    if fed_prox_mu:
        suffix += f", μ={fed_prox_mu}"

    VisualisationUtils.plot_metric_comparison(result_models, MetricType.ACCURACY,
                                              os.path.join(plot_dir, "accuracy.pdf"),
                                              f"[{privacy_method}] Test Accuracy – {suffix}",
                                              "Test Accuracy (%)", aggregation_method)

    VisualisationUtils.plot_metric_comparison(result_models, MetricType.LOSS,
                                              os.path.join(plot_dir, "loss.pdf"),
                                              f"[{privacy_method}] Test Loss – {suffix}",
                                              "Test Loss", aggregation_method)

    VisualisationUtils.plot_exec_times(list(result_models.values()),
                                       [VisualisationUtils.format_label(k, aggregation_method)
                                        for k in result_models],
                                       f"[{privacy_method}] Execution Time – {suffix}",
                                       os.path.join(plot_dir, "exec_time.pdf"))

    summary_path = os.path.join(os.path.dirname(plot_dir), "summary.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        for config_label, rows in summary_rows_by_config.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=config_label, index=False)
