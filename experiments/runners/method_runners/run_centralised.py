import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
import time

from common.const import CENTRALISED_EPOCHS
from common.model.model_wrapper import ModelWrapper
from trainers.centralised.centralised_trainer import CentralizedTrainer, CentralizedConfig
from common.result_utils.fl_result_utils import FLResultUtils

def run_centralized_experiment(
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_fn: Callable,
        dataset: str,
        epochs: int = CENTRALISED_EPOCHS,
        num_runs: int = 1,
        learning_rate: float = 1e-3,
        base_seed: int = 42
) -> ModelWrapper:
    print(f"\n[Centralized] Running {num_runs} run(s)")
    start = time.time()

    train_acc_runs, train_loss_runs = [], []
    test_acc_runs, test_loss_runs = [], []
    exec_times = []

    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        print(f"[Centralized] Run {run_idx + 1}/{num_runs} | seed = {run_seed}")

        torch.manual_seed(run_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)

        trainer = CentralizedTrainer(
            model_fn,
            train_loader,
            test_loader,
            CentralizedConfig(epochs=epochs, learning_rate=learning_rate)
        )
        model = trainer.train()
        train_acc_runs.append(model.train_acc)
        train_loss_runs.append(model.train_loss)
        test_acc_runs.append(model.test_acc)
        test_loss_runs.append(model.test_loss)
        exec_times.append(model.exec_time)

    def avg_metric(runs: list[list[float]]) -> list[float]:
        return [sum(xs) / len(xs) for xs in zip(*runs)]

    avg_model = ModelWrapper(
        model=model.model,
        train_acc=avg_metric(train_acc_runs),
        train_loss=avg_metric(train_loss_runs),
        test_acc=avg_metric(test_acc_runs),
        test_loss=avg_metric(test_loss_runs),
        exec_time=sum(exec_times) / num_runs
    )

    FLResultUtils.save(avg_model, dataset=dataset, method="centralized", aggregation="none")
    FLResultUtils.save_metadata(
        config={"type": "centralized"},
        dataset=dataset,
        method="centralized",
        aggregation="none"
    )

    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    print(f"Centralized training finished in {int(mins):02}:{int(secs):02} for {num_runs} run(s)")

    return avg_model

