from typing import List, Optional
from torch import nn


class ModelWrapper:
    def __init__(self, model: nn.Module, train_acc: List[float], train_loss: List[float], test_acc: List[float],
                 test_loss: List[float], exec_time: Optional[float] = 0) -> None:
        self.model = model
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss
        self.exec_time = exec_time