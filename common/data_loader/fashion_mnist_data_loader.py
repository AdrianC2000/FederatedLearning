from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from common.const import BATCH_SIZE, TEST_BATCH_SIZE
from common.data_loader.base_image_data_loader import BaseImageDataLoader


class FashionMNISTDataLoader:

    @staticmethod
    def get_dataloaders(batch_size: int = BATCH_SIZE,
                        test_batch_size: int = TEST_BATCH_SIZE,
                        subset_ratio: float = 1.0,
                        seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        project_root = Path(__file__).resolve().parents[1]
        data_root = project_root / "data"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6,), (0.6,))
        ])

        full_train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        full_test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

        train_dataset = BaseImageDataLoader().create_subset(full_train_dataset, subset_ratio, seed)
        test_dataset = BaseImageDataLoader().create_subset(full_test_dataset, subset_ratio, seed)

        print(f"[FashionMnist DataLoader] Train set size: {len(train_dataset)} samples")
        print(f"[FashionMnist DataLoader] Test set size:  {len(test_dataset)} samples")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader
