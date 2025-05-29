from enum import Enum

import numpy as np

from typing import Tuple, List

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
import torch


from sklearn.model_selection import StratifiedKFold

class DataSplitStrategy(Enum):
    STRATIFIED_EQUAL = "stratified_equal"
    STRATIFIED_IMBALANCED = "stratified_imbalanced"
    NON_IID_EQUAL = "non_iid_equal"

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

def split_dataset(
        dataset: Dataset,
        num_clients: int,
        data_split_strategy: DataSplitStrategy = DataSplitStrategy.STRATIFIED_EQUAL,
) -> List[Subset]:
    targets = np.array([sample[1] for sample in dataset])
    indices = np.arange(len(dataset))

    if data_split_strategy == DataSplitStrategy.STRATIFIED_EQUAL:
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True)
        splits = skf.split(indices, targets)
        return [Subset(dataset, list(idx)) for _, idx in splits]

    elif data_split_strategy == DataSplitStrategy.STRATIFIED_IMBALANCED:
        ratios = generate_balanced_imbalanced_ratios(num_clients, spread=1)
        total_samples = len(dataset)

        samples_per_client = (ratios * total_samples).astype(int)
        diff = total_samples - np.sum(samples_per_client)
        samples_per_client[0] += diff

        targets = np.array(targets)
        indices = np.arange(len(dataset))
        class_counts = np.bincount(targets)
        class_proportions = class_counts / total_samples

        per_client_indices = []

        for n_samples in samples_per_client:
            client_indices = []

            for cls, prop in enumerate(class_proportions):
                n_cls_samples = int(round(prop * n_samples))
                cls_indices = indices[targets == cls]
                np.random.shuffle(cls_indices)

                selected = cls_indices[:n_cls_samples]
                client_indices.extend(selected)

            np.random.shuffle(client_indices)
            client_indices = client_indices[:n_samples]
            per_client_indices.append(client_indices)

        return [Subset(dataset, idxs) for idxs in per_client_indices]

    elif data_split_strategy == DataSplitStrategy.NON_IID_EQUAL:
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients
        num_classes = np.max(targets) + 1

        class_to_indices = {cls: indices[targets == cls].tolist() for cls in range(num_classes)}
        for idxs in class_to_indices.values():
            np.random.shuffle(idxs)

        client_indices = [[] for _ in range(num_clients)]

        alpha = 0.5
        for cls, cls_idxs in class_to_indices.items():
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(cls_idxs)).astype(int)

            diff = len(cls_idxs) - np.sum(proportions)
            proportions[0] += diff

            start = 0
            for client_id, n in enumerate(proportions):
                client_indices[client_id].extend(cls_idxs[start:start + n])
                start += n

        all_assigned = set()
        final_indices = []
        for idxs in client_indices:
            np.random.shuffle(idxs)
            idxs = idxs[:samples_per_client]
            final_indices.append(idxs)
            all_assigned.update(idxs)

        remaining = list(set(indices) - all_assigned)
        np.random.shuffle(remaining)

        for i in range(num_clients):
            while len(final_indices[i]) < samples_per_client and remaining:
                final_indices[i].append(remaining.pop())

        return [Subset(dataset, idxs) for idxs in final_indices]
    return None

def generate_balanced_imbalanced_ratios(num_clients: int, spread: float = 0.3) -> np.ndarray:
    mid = num_clients // 2
    diffs = np.abs(np.arange(num_clients) - mid)
    max_val = diffs.max() or 1
    factors = 1 + (1 - diffs / max_val) * spread
    return factors / factors.sum()

def generate_class_distributions(num_clients: int, num_classes: int, alpha: float = 0.5) -> np.ndarray:
    return np.random.dirichlet([alpha] * num_classes, size=num_clients)