# Federated Learning with Privacy-Preserving Techniques

This repository implements a comprehensive framework for conducting federated learning (FL) experiments with support for multiple datasets, aggregation algorithms, and privacy-preserving methods. The project is part of a master's thesis on integrating privacy into federated training pipelines.

## Project Overview

- **Federated learning algorithms**: FedAvg, FedSGD, FedProx, FedAdam, FedAdagrad, FedYogi
- **Privacy methods**: Differential Privacy (DP), Homomorphic Encryption (HE), Secure Aggregation (SA), Secure Multi-Party Computation (SMPC)
- **Supported datasets**: Brain Tumor MRI, Chest X-Ray, Lung & Colon Cancer, FashionMNIST
- **Flexible configuration**: number of clients, local epochs, aggregation method, and privacy parameters

## Directory Structure

- `common/` – shared utilities, data loaders, enums, model definitions
- `data/` – local dataset folders (auto-downloaded)
- `experiments/` – scripts to run experiments for each dataset and method
- `trainers/` – centralized and federated training logic (with and without privacy)

## Datasets

The framework includes support for:

- **Brain Tumor MRI** (`brain_tumor`)
- **Chest X-Ray (Covid-19 & Pneumonia)** (`chest_xray`)
- **Lung & Colon Cancer Histopathology** (`lung_cancer`)

Custom data loaders are defined in `common/data_loader/` and handle downloading, preprocessing, and caching.

## Training Modes

### Centralized Learning
Standard training on all data without federation or privacy.

### Federated Plain (No Privacy)
- FedAvg
- FedSGD
- FedProx (μ)
- FedAdam / FedYogi / FedAdagrad

### Differential Privacy (DP)
- Based on Opacus
- Configurable ε, δ, gradient norm clipping
- Applied per-client during local training

### Homomorphic Encryption (HE)
- Based on TenSEAL
- Clients send encrypted model updates
- Server aggregates without decryption

### Secure Aggregation (SA)
- Clients mask model updates before sending
- Pairwise mask exchange (fully connected)
- Optionally simulates client dropout

### Secure Multi-Party Computation (SMPC)
- Each client splits its update into additive shares
- Shares are masked with noise
- Server aggregates without access to raw updates

## Aggregation Methods

Each method supports:

- **FedAvg** – weighted averaging
- **FedSGD** – server-side SGD on averaged gradients
- **FedProx** – penalizes divergence from global model
- **FedAdam, FedYogi, FedAdagrad** – server-side adaptive optimizers

All logic is handled in `trainers/<privacy_method>/federated_trainer.py`.

## Running Experiments

Each dataset has its own runner:

```
python experiments/dataset_runners/run_brain_tumor_experiments.py
python experiments/dataset_runners/run_chest_xray_experiments.py
python experiments/dataset_runners/run_lung_cancer_experiments.py
```

To run all experiments with all configurations:

```
python experiments/runners/all_experiments_runners.py
```

You can configure:
- `num_clients`, `num_rounds`, `local_epochs`
- `learning_rate`, `subset_ratio`
- `DataSplitStrategy`: `STRATIFIED_EQUAL`, `STRATIFIED_IMBALANCED`, `NON_IID_EQUAL`

## Output Format

Each experiment saves:

- `model.pt` – trained model (wrapped)
- `metrics.csv` – test accuracy/loss per round
- `metadata.csv` – full config
- `summary.xlsx` – summary for each configuration
- `plots/` – accuracy/loss/execution time charts

Structured under:

```
results/<dataset_timestamp>/<method>/<aggregation>/<config>/
```

## Extensibility

You can easily add:

- **New dataset**: define a loader in `data_loader`, register it in `get_dataloaders()`
- **New privacy method**: implement a `ClientTrainer` and `FederatedTrainer` with a config class
- **New aggregation method**: extend `AggregationMethod` enum and implement aggregation logic in each `FederatedTrainer`