import os
import random
import numpy as np
import pandas as pd
from typing import Optional

import torch
from torch.utils.data import DataLoader

from common.fml_utils import DataSplitStrategy
from trainers.federated_dp.client_trainer import DPConfig
from trainers.federated_dp.federated_trainer import FederatedTrainer, FederatedConfig
from common.result_utils.fl_result_utils import FLResultUtils
from common.result_utils.visualisation_utils import VisualisationUtils
from common.enum.aggregation_method import AggregationMethod
from common.enum.metric_type import MetricType
from common.model.model_wrapper import ModelWrapper


def run_dp_experiments(
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
        seed: int = 42,
):
    method = "differential_privacy"
    privacy_method = "Differential Privacy"
    aggregation = aggregation_method.value.lower()
    # epsilons = [0.1, 1.0, 3.0]
    # deltas = [1e-4, 0.1]
    epsilons = [1.0]
    deltas = [1e-4]

    relevant_baselines = {
        label: model for label, model in baseline_results.items()
        if label == "centralized" or label == aggregation
    }

    if aggregation_method == AggregationMethod.FED_PROX and fed_prox_mu is not None:
        mu_key = f"mu_{str(fed_prox_mu).replace('.', '_')}"
        relevant_baselines = {
            label: model for label, model in baseline_results.items()
            if label == "centralized" or label == mu_key
        }

    result_models = relevant_baselines.copy()

    summary_rows_by_config = {}

    def avg_metric(runs: list[list[float]]) -> list[float]:
        return [sum(xs) / len(xs) for xs in zip(*runs)]

    configs = [(e, d) for e in epsilons for d in deltas]
    for i, (eps, delta) in enumerate(configs, start=1):
        eps_str = f"{eps:.1f}".replace(".", "_")
        delta_str = f"{delta:.1e}".replace("e+0", "e").replace("e+", "e").replace(".0", "")
        config_label = f"e{eps_str}_d{delta_str}"
        print(f"[{i}/{len(configs)}] Running DP config: {config_label}")

        mu_key = f"mu_{str(fed_prox_mu).replace('.', '_')}" if fed_prox_mu is not None else None
        key = f"{mu_key}__{config_label}" if mu_key else config_label

        test_acc_runs, test_loss_runs = [], []
        exec_times = []

        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            print(f"[DP] Run {run_idx + 1}/{num_runs} | seed = {run_seed}")

            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            random.seed(run_seed)
            trainer = FederatedTrainer(
                train_loader,
                test_loader,
                model_fn,
                FederatedConfig(
                    dp_config=DPConfig(epsilon=eps, delta=delta),
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

        result_models[key] = avg_model

        FLResultUtils.save(avg_model, dataset, method, aggregation, key, fed_prox_mu)
        FLResultUtils.save_metadata(
            config={"epsilon": eps, "delta": delta, **({"fed_prox_mu": fed_prox_mu} if fed_prox_mu else {})},
            dataset=dataset,
            method=method,
            aggregation=aggregation,
            config_name=key,
            mu=fed_prox_mu,
        )

        summary_rows = []
        for r, (acc, loss) in enumerate(zip(avg_model.test_acc, avg_model.test_loss), start=1):
            summary_rows.append({
                "epsilon": eps,
                "delta": delta,
                "round": r,
                "test_acc": acc,
                "test_loss": loss,
                "exec_time": avg_model.exec_time,
            })

        summary_rows_by_config[key] = summary_rows

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
