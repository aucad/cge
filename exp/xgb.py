import sys
from os import devnull

# noinspection PyPackageRequirements
from art.estimators.classification import XGBoostClassifier
from xgboost import DMatrix, train as xg_train

from exp import BaseModel


class XGBoost(BaseModel):

    def __init__(self, conf):
        super().__init__(conf)
        self.name = 'XGBoost'

    def predict(self, x, y):
        tmp = self.model.predict(DMatrix(x, y))
        return tmp.argmax(axis=1 if len(tmp.shape) == 2 else 0)

    def train_steps(self):
        t_args, params = self.conf['train'], self.conf['params']
        d_train = DMatrix(self.train_x, self.train_y)
        sys.stdout = open(devnull, 'w')  # hide print
        self.model = xg_train(
            dtrain=d_train, **(t_args or {}),
            evals=[(d_train, 'eval'), (d_train, 'train')],
            params={**(params or {}), 'num_class': self.n_classes})
        sys.stdout = sys.__stdout__  # re-enable print
        self.classifier = XGBoostClassifier(
            model=self.model,
            clip_values=(0, 1),
            nb_features=self.train_x.shape[1],
            nb_classes=self.n_classes)
