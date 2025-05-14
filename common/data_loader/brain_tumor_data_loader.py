import shutil
import kagglehub
from torchvision import datasets

from common.data_loader.base_image_data_loader import BaseImageDataLoader


class BrainTumorDataLoader(BaseImageDataLoader):
    IMAGE_SIZE = 32
    GRAYSCALE = True
    DATA_FOLDER = "brain_tumor"

    def download_if_needed(self):
        expected_path = self.data_root / "brain_tumor_dataset"
        if not expected_path.exists():
            print("[BrainTumor] Downloading via kagglehub...")
            cache_path = kagglehub.dataset_download("rm1000/brain-tumor-mri-scans")
            shutil.move(cache_path, expected_path)

    def _load_dataset(self, transform):
        cache_path = self.data_root / f"cached_dataset_{self.IMAGE_SIZE}x{self.IMAGE_SIZE}_gray.pt"
        dataset_path = self.data_root / "brain_tumor_dataset"
        raw_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        return self.get_or_create_tensor_cache(raw_dataset, cache_path)
