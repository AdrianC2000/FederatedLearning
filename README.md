# Federated Learning with Privacy-Preserving Techniques

This repository implements a comprehensive framework for conducting federated learning (FL) experiments with support for multiple datasets, aggregation algorithms, and privacy-preserving methods. The project is part of a master's thesis on integrating privacy into federated training pipelines.

## Project Overview

- **Federated learning algorithms**: FedAvg, FedSGD, FedProx, FedAdam, FedAdagrad, FedYogi
- **Privacy methods**: Differential Privacy (DP), Homomorphic Encryption (HE), Secure Aggregation (SA), Secure Multi-Party Computation (SMPC)
- **Supported datasets**:  
  - [Brain Tumor MRI](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)  
  - [Chest X-Ray](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)  
  - [Lung & Colon Cancer](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-image)
- **Flexible configuration**:
  - **Number of clients** – specify the total number of participating clients (e.g., 10, 50, 100)
  - **Local epochs** – number of local training epochs before aggregation (e.g., 1–10)
  - **Batch size** – size of the training batches on the client side
  - **Learning rate** – learning rate used by the local optimizer (e.g., 0.001)
  - **Dataset split strategy**
    Specifies how the dataset is partitioned among clients. This setting affects the statistical distribution of data per client and can significantly influence the difficulty and realism of federated learning       scenarios.
    - **`STRATIFIED_EQUAL`**  
      Each client receives an equal number of samples, with label distribution matching the overall dataset (stratified). Ensures statistical homogeneity across clients.
    - **`STRATIFIED_IMBALANCED`**  
      Clients receive varying amounts of data, but label distribution within each client is still stratified. Useful for simulating real-world imbalance in data volume while maintaining label diversity.
    - **`NON_IID_EQUAL`**  
      Each client receives an equal number of samples, but with a skewed label distribution (non-iid). Typically, clients are biased toward specific classes, reflecting real-world scenarios where clients see           only a subset of data types.
  - **Random seed** – set seed for reproducibility of experiments
  - **Aggregation method** – choose the aggregation strategy, such as `FedAvg`, `FedProx`
  - **Privacy preserving methods configuration**:
    - **Differential Privacy** – configurable noise level (`ε`, `δ`, `max_grad_norm`), implemented using Opacus
    - **Homomorphic Encryption** – configurable encryption configuration (`scale`, `poly_modulus_degree`, `coeff_mod_bit_sizes`, `encryption_scheme`), implemented with TenSEAL
    - **Secure Aggregation** – `mask_noise_scale` parameter - standard deviation of the Gaussian noise added to the local model update (mask) + `drop_clients` parameter for checking the influence of lost client        in the training process
    - **Secure Multi-Party Computation** – `share_noise_scale` parameter - standard deviation of the Gaussian noise added during the simulated weight-sharing phase between clients + `drop_clients` parameters as        above
   
## Sample project results
You can see the sample results under `experiments/results` - these are the results for the `STRATIFIED_EQUAL` dataset split strategy.
Some of the generated plots are presented below:

## Directory Structure

- `common/` – shared utilities, data loaders, enums, model definitions
- `data/` – local dataset folders (auto-downloaded)
- `experiments/` – scripts to run experiments for each dataset and method
- `trainers/` – centralized and federated training logic (with and without privacy)

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
