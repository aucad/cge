from abc import ABC, abstractmethod

from exp import ModelScore


class ModelTraining(ABC):

    def __init__(self, conf):
        self.name = None
        self.classifier = None
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.score = None
        self.conf = conf or {}

    def reset(self, x, y, fold_train, fold_test):
        self.classifier = None
        self.model = None
        self.train_x = x[fold_train, :]
        self.train_y = y[fold_train].astype(int).flatten()
        self.test_x = x[fold_test, :]
        self.test_y = y[fold_test].astype(int).flatten()
        self.score = ModelScore()
        return self

    @property
    def n_classes(self):
        return len(list(set(list(self.train_y))))

    @abstractmethod
    def predict(self, x, y):
        pass

    @abstractmethod
    def train(self):
        pass
