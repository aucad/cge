import sys
import os

import numpy as np
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
        self.cls_conf = conf or {}

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
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def __train_classifier(self) -> None:
        d_train = DMatrix(self.train_x, self.train_y)
        n_classes = len(np.unique(self.train_y))
        sys.stdout = open(os.devnull, 'w')  # hide print
        self.model = xg_train(
            num_boost_round=10,
            dtrain=d_train,
            evals=[(d_train, 'eval'), (d_train, 'train')],
            params={'num_class': n_classes, **self.cls_conf})
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.train_x.shape[1],
            nb_classes=n_classes,
            clip_values=(0, 1))

    def train(self):
        """Train and score the model."""
        self.__train_classifier()
        self.score.calculate(
            self.test_y, self.predict(self.test_x, self.test_y))
        return self
