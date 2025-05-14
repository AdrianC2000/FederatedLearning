from enum import Enum

class Dataset(str, Enum):
    FASHION_MNIST = "fashionmnist"
    CHEST_XRAY = "chest_xray"
    LUNG_CANCER = "lung_cancer"
    BRAIN_TUMOR = "brain_tumor"
