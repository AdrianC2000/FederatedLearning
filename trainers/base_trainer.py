from abc import ABC, abstractmethod
from common.model.model_wrapper import ModelWrapper


class BaseTrainer(ABC):
    @abstractmethod
    def train(self) -> ModelWrapper:
        pass
