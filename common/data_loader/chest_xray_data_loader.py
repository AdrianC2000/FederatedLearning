import shutil
import kagglehub
import torch
from torchvision import datasets

from common.data_loader.base_image_data_loader import BaseImageDataLoader


class ChestXRayDataLoader(BaseImageDataLoader):
    IMAGE_SIZE = 32
    GRAYSCALE = True
    DATA_FOLDER = "chest_xray"

    def download_if_needed(self):
        expected_path = self.data_root / "Chest X-Ray (Covid-19 & Pneumonia)"
        if not expected_path.exists():
            print("[ChestXRay] Downloading via kagglehub...")
            cache_path = kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")
            shutil.move(cache_path, expected_path)

    def _load_dataset(self, transform):
        data_dir = self.data_root / "Chest X-Ray (Covid-19 & Pneumonia)" / "Data"
        train_cache = self.data_root / f"train_cached_{self.IMAGE_SIZE}x{self.IMAGE_SIZE}.pt"
        test_cache = self.data_root / f"test_cached_{self.IMAGE_SIZE}x{self.IMAGE_SIZE}.pt"

        train_raw = datasets.ImageFolder(root=data_dir / "train", transform=transform)
        test_raw = datasets.ImageFolder(root=data_dir / "test", transform=transform)

        train = self.get_or_create_tensor_cache(train_raw, train_cache)
        test = self.get_or_create_tensor_cache(test_raw, test_cache)

        return torch.utils.data.ConcatDataset([train, test])
