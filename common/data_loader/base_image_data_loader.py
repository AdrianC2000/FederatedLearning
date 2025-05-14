from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import shutil

class BaseImageDataLoader(ABC):
    IMAGE_SIZE: int
    GRAYSCALE: bool = True
    DATA_FOLDER: str

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_root = self.project_root / "data" / self.DATA_FOLDER

    @abstractmethod
    def download_if_needed(self): pass

    @abstractmethod
    def _load_dataset(self, transform) -> Dataset: pass

    def preprocess(self):
        tfms = [transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE))]
        if self.GRAYSCALE:
            tfms.insert(0, transforms.Grayscale(num_output_channels=1))
        tfms += [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        return transforms.Compose(tfms)

    def get_dataloaders(self, batch_size: int = 64, test_batch_size: int = 64,
                        subset_ratio: float = 1.0, seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        self.download_if_needed()
        dataset = self._load_dataset(self.preprocess())
        train_ds, test_ds = self._train_test_split(dataset, subset_ratio, seed)

        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True),
            DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        )

    def _train_test_split(self, dataset: Dataset, ratio: float, seed: Optional[int]):
        targets = [s[1] for s in dataset]
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=seed, stratify=targets)
        train = self.create_subset(Subset(dataset, train_idx), ratio, seed)
        test = self.create_subset(Subset(dataset, test_idx), ratio, seed)
        return train, test

    @staticmethod
    def create_subset(dataset, ratio: float, seed: Optional[int]):
        if ratio >= 1.0: return dataset
        if seed is not None: torch.manual_seed(seed)
        indices = torch.randperm(len(dataset))[:int(len(dataset) * ratio)]
        return Subset(dataset, indices)

    @staticmethod
    def get_or_create_tensor_cache(dataset: Dataset, cache_path: Path) -> TensorDataset:
        if cache_path.exists():
            print(f"[Cache] Loading: {cache_path}")
            data, labels = torch.load(cache_path)
        else:
            print(f"[Cache] Creating cache: {cache_path}")
            data, labels = zip(*[(img, label) for img, label in tqdm(dataset)])
            data = torch.stack(data)
            labels = torch.tensor(labels)
            torch.save((data, labels), cache_path)
            print(f"[Cache] Saved: {data.shape}, {labels.shape}")
        return TensorDataset(data, labels)

    @staticmethod
    def build_combined_imagefolder_structure(dest_root: Path, class_map: dict[str, Path]):
        for class_name, src_dir in class_map.items():
            dest = dest_root / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in src_dir.glob("*.*"):
                if f.is_file(): shutil.copy(f, dest / f.name)
