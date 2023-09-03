from abc import ABC, abstractmethod

from exp import ModelScore


class BaseModel(ABC):

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
        return max(2, len(list(set(list(self.train_y)))))

    def train(self):
        self.train_steps()
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        return self

    def to_dict(self):
        return {'name': self.name, 'config': self.conf}

    @abstractmethod
    def predict(self, x, y):
        pass

    @abstractmethod
    def train_steps(self):
        pass
