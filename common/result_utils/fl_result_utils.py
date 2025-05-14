import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from common.fml_utils import split_dataset_stratified
from common.model.model_wrapper import ModelWrapper
from pathlib import Path
import inspect

from torch.utils.data import Dataset
from collections import Counter
import csv


class FLResultUtils:
    @staticmethod
    def _build_base_path(
            dataset: str,
            method: str,
            aggregation: str,
            config: str = "",
            mu: float = None,
            client_group: str = None,  # "all_clients" | "dropped_client"
    ) -> str:
        parts = ["results", dataset, method]

        if aggregation != "none":
            parts.append(aggregation)

        if aggregation == "fed_prox" and mu is not None:
            parts.append(f"mu_{str(mu).replace('.', '_')}")

        if client_group:
            parts.append(client_group)

        if config and config != "default":
            parts.append(config)

        return os.path.join(*parts)

    @staticmethod
    def save(
            model: ModelWrapper,
            dataset: str,
            method: str,
            aggregation: str,
            config: str = "",
            mu: float = None,
            client_group: str = None,
    ):
        base_path = FLResultUtils._build_base_path(dataset, method, aggregation, config, mu, client_group)
        os.makedirs(base_path, exist_ok=True)

        torch.save(model, os.path.join(base_path, "model.pt"))

        rounds = list(range(1, len(model.test_acc) + 1))
        metrics_df = pd.DataFrame({
            "round": rounds,
            "test_acc": model.test_acc,
            "test_loss": model.test_loss,
        })
        metrics_df.to_csv(os.path.join(base_path, "metrics.csv"), index=False)


    @staticmethod
    def save_metadata(
            config: dict,
            dataset: str,
            method: str,
            aggregation: str,
            config_name: str = "",
            mu: float = None,
            client_group: str = None,
    ):
        base_path = FLResultUtils._build_base_path(dataset, method, aggregation, config_name, mu, client_group)
        pd.DataFrame([config]).to_csv(os.path.join(base_path, "metadata.csv"), index=False)

    @staticmethod
    def _plot_path(
            dataset: str,
            method: str,
            aggregation: str,
            config: str,
            mu: float = None,
            client_group: str = None,
    ) -> str:
        return os.path.join(
            FLResultUtils._build_base_path(dataset, method, aggregation, config, mu, client_group),
            "plots"
        )

    @staticmethod
    def load(name: str, dataset: str, out_dir: str = "saved_results") -> ModelWrapper:
        full_path = Path(out_dir) / dataset / name / "model.pt"
        if not full_path.exists():
            raise FileNotFoundError(f"No model found at {full_path.resolve()}")

        model_wrapper: ModelWrapper = torch.load(full_path, weights_only=False)
        print(f"Loaded model from: {full_path.resolve()}")
        return model_wrapper

    @staticmethod
    def _get_default_name_from_context() -> str:
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        return module.__name__.replace(".", "_") if module else "results"

    def write_config(
            self,
            output_dir: str,
            dataset: str,
            num_clients: int,
            num_rounds: int,
            local_epochs: int,
            learning_rate: float,
            num_runs: int,
            base_seed: int,
            subset_ratio: float,
            timestamp: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            model_fn: callable,
    ):
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.txt")
        model_name = model_fn().__class__.__name__
        with open(config_path, "w") as f:
            f.write("Federated Experiment Configuration\n")
            f.write("=" * 40 + "\n")
            f.write(f"Dataset        : {dataset.split('_')[0]}\n")
            f.write(f"Model          : {model_name}\n")
            f.write(f"Num Clients    : {num_clients}\n")
            f.write(f"Num Rounds     : {num_rounds}\n")
            f.write(f"Local Epochs   : {local_epochs}\n")
            f.write(f"Learning Rate  : {learning_rate}\n")
            f.write(f"Num Runs       : {num_runs}\n")
            f.write(f"Base Seed      : {base_seed}\n")
            f.write(f"Subset Ratio   : {subset_ratio}\n")
            f.write(f"Train Size     : {len(train_loader.dataset)}\n")
            f.write(f"Test Size      : {len(test_loader.dataset)}\n")
            f.write(f"Timestamp      : {timestamp}\n")

        num_classes = max(label for _, label in train_loader.dataset) + 1
        class_names = [f"class_{i}" for i in range(num_classes)]

        self.log_data_distribution(
            dataset=train_loader.dataset,
            num_clients=num_clients,
            num_runs=num_runs,
            base_seed=base_seed,
            output_path=os.path.join(output_dir, "client_data_distribution.csv"),
            class_names=class_names
        )

    @staticmethod
    def log_data_distribution(
            dataset: Dataset,
            num_clients: int,
            num_runs: int,
            base_seed: int,
            output_path: str,
            class_names: list[str]
    ):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "client_id", "total"] + class_names)

            for run_idx in range(num_runs):
                seed = base_seed + run_idx
                client_datasets = split_dataset_stratified(dataset, num_clients, seed)

                for i, subset in enumerate(client_datasets):
                    labels = [int(label) for _, label in subset]
                    label_counts = Counter(labels)
                    row = [run_idx, i, len(subset)] + [label_counts.get(j, 0) for j in range(len(class_names))]
                    writer.writerow(row)
