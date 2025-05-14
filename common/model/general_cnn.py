import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeneralCNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 32)

        self.pool = nn.MaxPool2d(2, 2)

        conv_output = input_size // 4
        flattened = 32 * conv_output * conv_output

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(flattened, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
