from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class FashionMNISTDataLoader:
    @staticmethod
    def get_dataloaders(batch_size: int = 32, test_batch_size: int = 1000) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6,), (0.6,))
        ])

        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader
