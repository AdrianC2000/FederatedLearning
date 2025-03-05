from abc import ABC, abstractmethod

from common.model_wrapper import ModelWrapper


class Trainer(ABC):
    @abstractmethod
    def train(self) -> ModelWrapper:
        pass
