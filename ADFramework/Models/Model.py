from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, x_train, x_val, training_config, early_stopping):
        pass

    @abstractmethod
    def predict(self, x, reconstruct=True):
        pass

    @abstractmethod
    def save(self, save_dir, file_name):
        pass

    @abstractmethod
    def load(cls, model_dir, file_name):
        pass
