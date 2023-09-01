import sys
from os import devnull

from art.estimators.classification import XGBoostClassifier
from xgboost import DMatrix, train as xg_train

from exp import TargetModel


class XGBoost(TargetModel):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = 'XGBoost'

    def predict(self, x, y):
        tmp = self.model.predict(DMatrix(x, y))
        return tmp.argmax(axis=1 if len(tmp.shape) == 2 else 0)

    def train(self):
        d_train = DMatrix(self.train_x, self.train_y)
        sys.stdout = open(devnull, 'w')  # hide print
        rest = dict([x for x in self.conf.items() if x[0] != 'params'])
        params = self.conf['params'] if 'params' in self.conf else {}
        self.model = xg_train(
            evals=[(d_train, 'eval'), (d_train, 'train')],
            dtrain=d_train,
            params={**params, 'num_class': self.n_classes},
            **rest)
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model,
            **{'clip_values': (0, 1),
               'nb_features': self.train_x.shape[1],
               'nb_classes': self.n_classes})
        predictions = self.predict(self.test_x, self.test_y)
        self.score.calculate(self.test_y, predictions)
        return self
