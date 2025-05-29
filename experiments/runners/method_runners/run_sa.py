import os
import pandas as pd
import torch
import random
import numpy as np
from typing import Optional

from torch.utils.data import DataLoader

from common.fml_utils import DataSplitStrategy
from trainers.federated_sa.client_trainer import SAConfig
from trainers.federated_sa.federated_trainer import FederatedTrainer, FederatedConfig
from common.result_utils.fl_result_utils import FLResultUtils
from common.result_utils.visualisation_utils import VisualisationUtils
from common.enum.aggregation_method import AggregationMethod
from common.enum.metric_type import MetricType
from common.model.model_wrapper import ModelWrapper


def run_sa_experiments(
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
        data_split_strategy: DataSplitStrategy,
        fed_prox_mu: Optional[float] = None,
        seed: int = 42
):
    method = "secure_aggregation"
    privacy_method = "Secure Aggregation"
    aggregation = aggregation_method.value.lower()
    noise_scales = [0.1, 0.25, 0.5]

    drop_clients_values = [False, True] if data_split_strategy == DataSplitStrategy.STRATIFIED_EQUAL else [False]

    for drop_clients in drop_clients_values:
        client_group = "dropped_client" if drop_clients else "all_clients"
        if aggregation_method == AggregationMethod.FED_PROX and fed_prox_mu is not None:
            mu_key = f"mu_{str(fed_prox_mu).replace('.', '_')}"
            result_models = {
                label: model for label, model in baseline_results.items()
                if label == "centralized" or label == mu_key
            }
        else:
            result_models = {
                label: model for label, model in baseline_results.items()
                if label == "centralized" or label == aggregation
            }

        result_models_by_mu = {
            "mu_0_001": result_models.copy(),
            "mu_0_01": result_models.copy(),
            "mu_0_1": result_models.copy(),
        } if aggregation_method == AggregationMethod.FED_PROX else {}

        summary_rows_by_config = {}

        def avg_metric(runs: list[list[float]]) -> list[float]:
            return [sum(xs) / len(xs) for xs in zip(*runs)]

        for noise in noise_scales:
            config_label = f"mask_noise_scale_{str(noise).replace('.', '_')}"
            print(f"→ Running SA config: {config_label} | {client_group.upper()}")

            test_acc_runs, test_loss_runs, exec_times = [], [], []

            for run_idx in range(num_runs):
                run_seed = seed + run_idx
                print(f"[SA] Run {run_idx + 1}/{num_runs} | seed = {run_seed}")

                torch.manual_seed(run_seed)
                random.seed(run_seed)
                np.random.seed(run_seed)

                trainer = FederatedTrainer(
                    train_loader,
                    test_loader,
                    model_fn,
                    FederatedConfig(
                        sa_config=SAConfig(mask_noise_scale=noise, drop_clients=drop_clients),
                        num_clients=num_clients,
                        num_rounds=num_rounds,
                        local_epochs=local_epochs,
                        learning_rate=learning_rate,
                        aggregation_method=aggregation_method,
                        fed_prox_mu=fed_prox_mu,
                        data_split_strategy=data_split_strategy
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

            FLResultUtils.save(avg_model, dataset, method, aggregation, config_label, fed_prox_mu, client_group)
            FLResultUtils.save_metadata(
                config={
                    "mask_noise_scale": noise,
                    "drop_clients": drop_clients,
                    **({"fed_prox_mu": fed_prox_mu} if fed_prox_mu else {})
                },
                dataset=dataset,
                method=method,
                aggregation=aggregation,
                config_name=config_label,
                mu=fed_prox_mu,
                client_group=client_group,
            )

            summary_rows = []
            for r, (acc, loss) in enumerate(zip(avg_model.test_acc, avg_model.test_loss), start=1):
                summary_rows.append({
                    "mask_noise_scale": noise,
                    "round": r,
                    "test_acc": acc,
                    "test_loss": loss,
                    "exec_time": avg_model.exec_time,
                })

            summary_rows_by_config[config_label] = summary_rows

        summary_dir = os.path.join("results", dataset, method, aggregation,
                                   *(["mu_" + str(fed_prox_mu).replace(".", "_")] if fed_prox_mu else []),
                                   client_group)
        plot_dir = os.path.join(summary_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        suffix = VisualisationUtils.get_aggregation_title(aggregation_method)
        if fed_prox_mu:
            suffix += f", μ={fed_prox_mu}"
        suffix += " (One Client Dropped)" if drop_clients else " (All Clients)"

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

        summary_path = os.path.join(summary_dir, "summary.xlsx")
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            for config_label, rows in summary_rows_by_config.items():
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=config_label, index=False)

        print(f"[SA] Saved summary to {summary_path}")
