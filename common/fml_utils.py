from typing import Tuple

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
import torch


from sklearn.model_selection import StratifiedKFold

def remove_module_prefix(state_dict):
    """Remove prefix '_module.' added by Opacus."""
    return {k.replace("_module.", ""): v for k, v in state_dict.items()}

def compute_loss_and_accuracy(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item()

    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

def split_dataset_stratified(dataset: Dataset, num_clients: int, seed: int = 42) -> list[Subset]:
    targets = [sample[1] for sample in dataset]
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    indices = list(range(len(dataset)))
    splits = skf.split(indices, targets)
    return [Subset(dataset, list(idx)) for _, idx in splits]

