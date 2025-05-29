from common.data_loader import get_dataloaders
from common.enum.dataset import Dataset
from common.fml_utils import DataSplitStrategy
from common.model.lung_cancer_cnn import LungCancerCNN
from experiments.runners.federated_runner import run_all_experiments


def run_experiments_for_lung_cancer_dataset(data_split_strategy: DataSplitStrategy = DataSplitStrategy.STRATIFIED_EQUAL):
    dataset = Dataset.LUNG_CANCER
    model_fn = LungCancerCNN
    num_runs = 1
    subset_ratio = 0.5
    seed = 42
    num_clients = 5
    num_rounds = 10
    local_epochs = 1
    learning_rate = 0.01
    batch_size = 128
    test_batch_size = 128

    train_loader, test_loader = get_dataloaders(
        dataset=dataset,
        subset_ratio=subset_ratio,
        seed=seed,
        batch_size=batch_size,
        test_batch_size=test_batch_size
    )

    run_all_experiments(
        dataset=dataset.value,
        model_fn=model_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        num_runs=num_runs,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        base_seed=seed,
        subset_ratio=subset_ratio,
        data_split_strategy=data_split_strategy
    )

if __name__ == "__main__":
    run_experiments_for_lung_cancer_dataset()

