import os
import time
from datetime import datetime

from torch.utils.data import DataLoader

from common.enum.aggregation_method import AggregationMethod
from common.fml_utils import DataSplitStrategy
from common.model.model_wrapper import ModelWrapper
from common.result_utils.fl_result_utils import FLResultUtils
from experiments.runners.method_runners.run_centralised import run_centralized_experiment
from experiments.runners.method_runners.run_plain import run_plain_experiments

from experiments.runners.method_runners.run_dp import run_dp_experiments
from experiments.runners.method_runners.run_he import run_he_experiments
from experiments.runners.method_runners.run_sa import run_sa_experiments
from experiments.runners.method_runners.run_smpc import run_smpc_experiments

ALL_RUNNERS = [run_dp_experiments, run_he_experiments, run_sa_experiments, run_smpc_experiments]

def print_time(label: str, start: float, global_start: float):
    local_elapsed = time.time() - start
    global_elapsed = time.time() - global_start

    def format_sec(seconds):
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}:{int(m):02}:{int(s):02}"

    print(f"{label} finished in {format_sec(local_elapsed)} (total time: {format_sec(global_elapsed)})")


def run_fedavg_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                           runners: list):
    print("\n[Running FedAvg experiments]")
    fedavg_start = time.time()
    for runner in runners:
        method_start = time.time()
        runner(aggregation_method=AggregationMethod.FED_AVG, baseline_results=baseline_results, **shared_args)
        print_time(f"{runner.__name__} [FedAvg]", method_start, global_start)
    print_time("All FedAvg experiments", fedavg_start, global_start)


def run_fedsgd_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                           runners: list):
    print("\n[Running FedSGD experiments]")
    fedsgd_start = time.time()
    for runner in runners:
        method_start = time.time()
        runner(aggregation_method=AggregationMethod.FED_SGD, baseline_results=baseline_results, **shared_args)
        print_time(f"{runner.__name__} [FedSGD]", method_start, global_start)
    print_time("All FedSGD experiments", fedsgd_start, global_start)


def run_fedprox_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                            runners: list):
    print("\n[Running FedProx experiments]")
    fedprox_start = time.time()
    for mu in [1, 5, 10]:
        print(f"→ FedProx | mu = {mu}")
        prox_start = time.time()
        prox_args = {**shared_args, "fed_prox_mu": mu}
        for runner in runners:
            method_start = time.time()
            runner(aggregation_method=AggregationMethod.FED_PROX, baseline_results=baseline_results, **prox_args)
            print_time(f"{runner.__name__} [FedProx, μ={mu}]", method_start, global_start)
        print_time(f"FedProx μ={mu}", prox_start, global_start)
    print_time("All FedProx experiments", fedprox_start, global_start)


def run_fedadam_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                            runners: list):
    print("\n[Running FedAdam experiments]")
    fedadam_start = time.time()
    for runner in runners:
        method_start = time.time()
        runner(aggregation_method=AggregationMethod.FED_ADAM, baseline_results=baseline_results, **shared_args)
        print_time(f"{runner.__name__} [FedAdam]", method_start, global_start)
    print_time("All FedAdam experiments", fedadam_start, global_start)


def run_fedadagrad_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                               runners: list):
    print("\n[Running FedAdagrad experiments]")
    fedadagrad_start = time.time()
    for runner in runners:
        method_start = time.time()
        runner(aggregation_method=AggregationMethod.FED_ADAGRAD, baseline_results=baseline_results, **shared_args)
        print_time(f"{runner.__name__} [FedAdagrad]", method_start, global_start)
    print_time("All FedAdagrad experiments", fedadagrad_start, global_start)


def run_fedyogi_experiments(shared_args: dict, baseline_results: dict[str, ModelWrapper], global_start: float,
                            runners: list):
    print("\n[Running FedYogi experiments]")
    fedyogi_start = time.time()
    for runner in runners:
        method_start = time.time()
        runner(aggregation_method=AggregationMethod.FED_YOGI, baseline_results=baseline_results, **shared_args)
        print_time(f"{runner.__name__} [FedYogi]", method_start, global_start)
    print_time("All FedYogi experiments", fedyogi_start, global_start)


def run_all_experiments(
        dataset: str,
        model_fn: callable,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_runs: int,
        num_clients: int,
        num_rounds: int,
        local_epochs: int,
        learning_rate: float,
        base_seed: int,
        subset_ratio: float,
        data_split_strategy: DataSplitStrategy
):
    global_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = f"{dataset}_{timestamp}"

    FLResultUtils().write_config(
        output_dir=os.path.join("results", dataset),
        dataset=dataset,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        num_runs=num_runs,
        base_seed=base_seed,
        subset_ratio=subset_ratio,
        timestamp=timestamp,
        train_loader=train_loader,
        test_loader=test_loader,
        model_fn=model_fn,
        data_split_strategy=data_split_strategy
    )

    centralized = run_centralized_experiment(
        train_loader=train_loader,
        test_loader=test_loader,
        model_fn=model_fn,
        dataset=dataset,
        epochs=num_rounds,
        num_runs=num_runs,
        learning_rate=learning_rate,
        base_seed=base_seed
    )

    plain = run_plain_experiments(
        train_loader=train_loader,
        test_loader=test_loader,
        model_fn=model_fn,
        dataset=dataset,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        num_runs=num_runs,
        learning_rate=learning_rate,
        base_seed=base_seed,
        data_split_strategy=data_split_strategy
    )
    baseline_results = {"centralized": centralized, **plain}

    shared_args = dict(
        train_loader=train_loader,
        test_loader=test_loader,
        model_fn=model_fn,
        dataset=dataset,
        num_runs=num_runs,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        seed=base_seed,
        data_split_strategy=data_split_strategy
    )

    run_fedavg_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)
    run_fedsgd_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)
    run_fedprox_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)
    run_fedadam_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)
    run_fedadagrad_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)
    run_fedyogi_experiments(shared_args, baseline_results, global_start, ALL_RUNNERS)

    total_time = time.time() - global_start
    hrs, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nAll experiments finished in {int(hrs):02}:{int(mins):02}:{int(secs):02}")
