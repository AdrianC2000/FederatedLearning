import shutil
import kagglehub
from torchvision import datasets

from common.data_loader.base_image_data_loader import BaseImageDataLoader


class LungCancerDataLoader(BaseImageDataLoader):
    IMAGE_SIZE = 24
    GRAYSCALE = True
    DATA_FOLDER = "lung_cancer"

    def download_if_needed(self):
        expected_path = self.data_root / "lung-and-colon-cancer-histopathological-images"
        if not expected_path.exists():
            print("[LungCancer] Downloading via kagglehub...")
            cache_path = kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
            shutil.move(cache_path, expected_path)

        combined_path = self.data_root / "all_images"
        if not combined_path.exists():
            image_root = expected_path / "lung_colon_image_set"
            print("[LungCancer] Building imagefolder structure...")
            self.build_combined_imagefolder_structure(
                dest_root=combined_path,
                class_map={
                    "adenocarcinoma": image_root / "lung_image_sets" / "lung_aca",
                    "benign": image_root / "lung_image_sets" / "lung_n",
                    "squamous": image_root / "lung_image_sets" / "lung_scc",
                    "colon_adenocarcinoma": image_root / "colon_image_sets" / "colon_aca",
                    "colon_benign": image_root / "colon_image_sets" / "colon_n",
                }
            )

    def _load_dataset(self, transform):
        combined_path = self.data_root / "all_images"
        cache_path = self.data_root / f"cached_dataset_{self.IMAGE_SIZE}x{self.IMAGE_SIZE}_gray.pt"
        raw_dataset = datasets.ImageFolder(root=combined_path, transform=transform)
        return self.get_or_create_tensor_cache(raw_dataset, cache_path)
