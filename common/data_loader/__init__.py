from torch.utils.data import DataLoader

from .brain_tumor_data_loader import BrainTumorDataLoader
from .fashion_mnist_data_loader import FashionMNISTDataLoader
from .chest_xray_data_loader import ChestXRayDataLoader
from .lung_cancer_data_loader import LungCancerDataLoader

from ..enum.dataset import Dataset


def get_dataloaders(dataset: Dataset, subset_ratio: float, seed: int) -> tuple[DataLoader, DataLoader]:
    if dataset == Dataset.FASHION_MNIST:
        return FashionMNISTDataLoader.get_dataloaders(subset_ratio=subset_ratio, seed=seed)
    elif dataset == Dataset.CHEST_XRAY:
        return ChestXRayDataLoader().get_dataloaders(subset_ratio=subset_ratio, seed=seed)
    elif dataset == Dataset.LUNG_CANCER:
        return LungCancerDataLoader().get_dataloaders(subset_ratio=subset_ratio, seed=seed)
    elif dataset == Dataset.BRAIN_TUMOR:
        return BrainTumorDataLoader().get_dataloaders(subset_ratio=subset_ratio, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
