from common.data_loader import get_dataloaders
from common.enum.dataset import Dataset
from common.model.simple_cnn import SimpleCNN
from experiments.runners.federated_runner import run_all_experiments


def run_all():
    dataset = Dataset.FASHION_MNIST
    model_fn = SimpleCNN
    num_runs = 3
    subset_ratio = 0.1
    seed = 42
    num_clients = 5
    num_rounds = 10
    local_epochs = 1
    learning_rate = 0.01

    train_loader, test_loader = get_dataloaders(
        dataset=dataset,
        subset_ratio=subset_ratio,
        seed=seed
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
        subset_ratio=subset_ratio
    )

if __name__ == "__main__":
    run_all()

