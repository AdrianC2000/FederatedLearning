from typing import List
import matplotlib.pyplot as plt

from common.enum.metric_type import MetricType
from common.enum.aggregation_method import AggregationMethod
from common.model.model_wrapper import ModelWrapper


class VisualisationUtils:

    @staticmethod
    def plot_metric(
            models: List[ModelWrapper],
            labels: List[str],
            metric_type: MetricType,
            title: str = "",
            ylabel: str = ""
    ) -> None:
        plt.figure(figsize=(10, 6))

        for model, label in zip(models, labels):
            if metric_type == MetricType.ACCURACY:
                values = model.test_acc
                ylabel = ylabel or "Accuracy (%)"
            elif metric_type == MetricType.LOSS:
                values = model.test_loss
                ylabel = ylabel or "Loss"
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

            x_axis = list(range(1, len(values) + 1))
            plt.plot(x_axis, values, label=label, marker="o")

        plt.xlabel("Federated Rounds")
        plt.ylabel(ylabel)
        plt.title(title or f"Test {metric_type.value.capitalize()}")
        plt.legend()
        plt.grid(True)

        total_rounds = max(len(model.test_acc if metric_type == MetricType.ACCURACY else model.test_loss)
                           for model in models)
        if total_rounds <= 10:
            plt.xticks(range(1, total_rounds + 1))
        elif total_rounds <= 50:
            plt.xticks(range(1, total_rounds + 1, 5))
        else:
            plt.xticks(range(1, total_rounds + 1, 10))

        plt.show()

    @staticmethod
    def plot_exec_times(
            models: list[ModelWrapper],
            labels: list[str],
            title: str,
            out_path: str,
    ):
        times = [model.exec_time for model in models]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, times)
        plt.ylabel("Czas [s]")
        plt.title(title)
        plt.xticks(rotation=30)
        plt.ylim(top=max(times) * 1.15)
        plt.grid(True, axis="y")

        for bar, time in zip(bars, times):
            height = bar.get_height()
            if time >= 60:
                minutes = int(time // 60)
                seconds = int(time % 60)
                label = f"{minutes}min{seconds:02d}sek"
            else:
                label = f"{time:.2f}s"
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.3, label,
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def plot_metric_comparison(
            models: dict[str, ModelWrapper],
            metric_type: MetricType,
            out_path: str,
            title: str,
            ylabel: str,
            aggregation_method: AggregationMethod,
    ):
        plt.figure(figsize=(10, 6))
        max_len = max(len(model.test_acc if metric_type == MetricType.ACCURACY else model.test_loss)
                      for model in models.values())

        for label, model in models.items():
            y_vals = model.test_acc if metric_type == MetricType.ACCURACY else model.test_loss
            x_vals = list(range(1, len(y_vals) + 1))
            formatted_label = VisualisationUtils.format_label(label, aggregation_method)
            plt.plot(x_vals, y_vals, label=formatted_label, marker="o")

        plt.xlabel("Communication Round / Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="upper right")

        # Tick frequency logic
        if max_len <= 10:
            plt.xticks(range(1, max_len + 1))
        elif max_len <= 50:
            plt.xticks(range(1, max_len + 1, 5))
        else:
            plt.xticks(range(1, max_len + 1, 10))

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def format_label(label: str, aggregation_method: AggregationMethod) -> str:
        if label == "centralized":
            return "Centralized"

        if label == "fed_avg":
            return "FedAvg (No Privacy)"
        if label == "fed_sgd":
            return "FedSGD (No Privacy)"
        if label == "fed_prox":
            return "FedProx (No Privacy)"
        if label == "fed_adam":
            return "FedAdam (No Privacy)"
        if label == "fed_adagrad":
            return "FedAdagrad (No Privacy)"
        if label == "fed_yogi":
            return "FedYogi (No Privacy)"

        if label.startswith("mu_"):
            mu_val = label.split("_", 1)[1].replace("_", ".")
            if aggregation_method == AggregationMethod.FED_PROX:
                return f"FedProx Plain (μ = {mu_val})"
            if aggregation_method == AggregationMethod.FED_AVG:
                return f"FedAvg Plain (μ = {mu_val})"
            if aggregation_method == AggregationMethod.FED_SGD:
                return f"FedSGD Plain (μ = {mu_val})"
            return f"{aggregation_method.value.title()} Plain (μ = {mu_val})"

        if label.startswith("scale_2^"):
            return f"scale = {label.replace('scale_', '')}"

        if label.startswith("mask_noise_scale_"):
            val = label.replace("mask_noise_scale_", "").replace("_", ".")
            return f"Mask Noise = {val}"

        if label.startswith("share_noise_scale_"):
            val = label.replace("share_noise_scale_", "").replace("_", ".")
            return f"Share Noise = {val}"

        if label.startswith("e") and "_d" in label:
            try:
                eps_part = label.split("_d")[0][1:]
                delta_part = label.split("_d")[1]

                eps = float(eps_part.replace("_", "."))
                delta = float(delta_part.replace("_", "."))

                delta_str = f"{delta:.1e}".replace("e+0", "e").replace("e+", "e").replace(".0", "")

                return f"ε = {eps}, δ = {delta_str}"
            except Exception:
                return label

        return label

    @staticmethod
    def get_aggregation_title(aggregation_method: AggregationMethod) -> str:
        titles = {
            AggregationMethod.FED_AVG: "Federated Averaging",
            AggregationMethod.FED_SGD: "Federated SGD",
            AggregationMethod.FED_PROX: "Federated Prox",
            AggregationMethod.FED_ADAM: "Federated Adam",
            AggregationMethod.FED_ADAGRAD: "Federated Adagrad",
            AggregationMethod.FED_YOGI: "Federated Yogi"
        }
        return titles.get(aggregation_method, aggregation_method.value.title())
