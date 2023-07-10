import sys
import os

import numpy as np
from art.estimators.classification import XGBoostClassifier
from xgboost import DMatrix, train as xg_train

from exp import Utility as Util


class ModelTraining:

    def __init__(self, out, attrs, y, ranges, conf):
        self.name = 'xgboost'
        self.out_dir = out
        self.attrs = attrs[:]
        self.n_features = len(self.attrs) - 1
        self.classes = np.unique(y)
        self.attr_ranges = ranges
        self.classifier = None
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.fold_n = 1
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0
        self.reset()
        self.cls_conf = conf or {}

    def reset(self):
        self.classifier = None
        self.model = None
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.fold_n = 1
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f_score = 0
        return self

    @property
    def class_names(self):
        return [('malicious' if cn == 1 else 'benign')
                for cn in self.classes]

    def load(self, x, y, fold_train, fold_test, fold_n):
        self.train_x = self.normalize(x[fold_train, :])
        self.train_y = y[fold_train].astype(int).flatten()
        self.test_x = self.normalize(x[fold_test, :])
        self.test_y = y[fold_test].astype(int).flatten()
        self.classes = np.unique(y)
        self.fold_n = fold_n
        return self

    @staticmethod
    def formatter(x, y):
        return DMatrix(x, y)

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def init_learner(self) -> None:
        d_train = self.formatter(self.train_x, self.train_y)
        sys.stdout = open(os.devnull, 'w')
        self.model = xg_train(
            num_boost_round=20,
            dtrain=d_train,
            evals=[(d_train, 'eval'), (d_train, 'train')],
            params={'num_class': len(self.classes), **self.cls_conf})
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model,
            nb_features=self.n_features,
            nb_classes=len(self.classes),
            clip_values=(0, 1))

    def train(self):
        self.init_learner()
        records = (
            (self.test_x, self.test_y) if len(self.test_x) > 0
            else (self.train_x, self.train_y))
        predictions = self.predict(self.formatter(*records))
        self.score(records[1], predictions)
        return self

    def normalize(self, data):
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(self.n_features):
            range_max = self.attr_ranges[i]
            data[:, i] = (data[:, i]) / range_max
            data[:, i] = np.nan_to_num(data[:, i])
        return data

    def score(self, true_labels, predictions, positive=0):
        """Calculate performance metrics."""
        tp, tp_tn, p_pred, p_actual = 0, 0, 0, 0
        for actual, pred in zip(true_labels, predictions):
            int_pred = int(round(pred, 0))
            if int_pred == positive:
                p_pred += 1
            if actual == positive:
                p_actual += 1
            if int_pred == actual:
                tp_tn += 1
            if int_pred == actual and int_pred == positive:
                tp += 1
        self.accuracy = Util.sdiv(tp_tn, len(predictions), -1, False)
        self.precision = Util.sdiv(tp, p_pred, 1, False)
        self.recall = Util.sdiv(tp, p_actual, 1, False)
        self.f_score = Util.sdiv(
            2 * self.precision * self.recall,
            self.precision + self.recall, 0, False)
