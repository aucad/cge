import sys
from os import devnull

from art.estimators.classification import XGBoostClassifier
from xgboost import DMatrix, train as xg_train

from exp import ModelScore


class ModelTraining:
    """Wrapper for training XGBoost."""

    def __init__(self, conf):
        self.name = 'XGBoost'
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

    def predict(self, x, y):
        tmp = self.model.predict(DMatrix(x, y))
        return tmp.argmax(axis=1 if len(tmp.shape) == 2 else 0)

    @property
    def n_classes(self):
        return len(list(set(list(self.train_y))))

    @property
    def train_config(self):
        return {'num_boost_round': 10,
                'params': {'num_class': self.n_classes, **self.conf}}

    @property
    def cls_config(self):
        return {'clip_values': (0, 1),
                'nb_features': self.train_x.shape[1],
                'nb_classes': self.n_classes}

    def train(self):
        d_train = DMatrix(self.train_x, self.train_y)
        sys.stdout = open(devnull, 'w')  # hide print
        self.model = xg_train(
            evals=[(d_train, 'eval'), (d_train, 'train')],
            dtrain=d_train, **self.train_config)
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model, **self.cls_config)
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        return self
